"""Read-only static environment checks. Does not import third-party packages."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


MODULES = ("numpy", "cv2", "yaml", "onnxruntime", "PyQt6", "torch")


def inspect_model(model: str | None) -> dict:
    result = {"fileVerified": False, "loaded": False, "formatValidated": False,
              "runtimeReady": False, "path": model, "sha256": None}
    if not model:
        result["reason"] = "No model supplied. Public YOLO-omni weights have not been found."
        return result
    path = Path(model).expanduser().resolve()
    result["path"] = str(path)
    if path.suffix.lower() not in (".onnx", ".pt"):
        result["reason"] = "Only ONNX/PT paths are accepted; no executable files."
        return result
    try:
        if not path.is_file() or path.stat().st_size == 0:
            result["reason"] = "Model file is missing or empty."
            return result
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result.update(fileVerified=True, bytes=path.stat().st_size,
                      sha256=digest.hexdigest(),
                      reason="File exists and has a SHA256; model contents and provenance are not validated.")
    except OSError as exc:
        result["reason"] = str(exc)
    return result


def inspect_detector(repo_root: Path) -> dict:
    path = repo_root / "Desktop" / "models" / "detector.py"
    result = {"path": str(path.resolve()), "placeholder": None,
              "inferenceImplementationFound": False, "executed": False}
    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        methods = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "predict"]
        if not methods:
            result["reason"] = "No predict() implementation found."
            return result
        # Audit every predict definition; a random or stub branch disqualifies it.
        predictors = [ast.walk(method) for method in methods]
        nodes = [node for iterator in predictors for node in iterator]
        calls = [ast.unparse(node.func) for node in nodes if isinstance(node, ast.Call)]
        stub = any("random" in call.split(".") or call.endswith(".randn")
                   or call.endswith(".rand") for call in calls)
        stub |= any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "placeholder" in node.value.lower() for node in nodes)
        stub |= any(isinstance(node, ast.Name) and node.id == "NotImplementedError" for node in nodes)
        result["placeholder"] = bool(stub)
        real_call = any(call.endswith(".run") or call == "self.model" for call in calls)
        result["inferenceImplementationFound"] = bool(real_call and not stub)
        result["reason"] = ("Random/placeholder predict() is unusable." if stub else
                            "Static inference call found; execution and output contract remain untested." if real_call else
                            "No recognized inference call in predict(); review required.")
    except (OSError, SyntaxError, UnicodeError) as exc:
        result["reason"] = str(exc)
    return result


def verify(repo_root: Path, model: str | None, finder=importlib.util.find_spec,
           version=None) -> dict:
    version = tuple(version if version is not None else sys.version_info[:3])
    recommended = (3, 10) <= version[:2] <= (3, 12)
    dependencies = {}
    for name in MODULES:
        try:
            dependencies[name] = {"found": finder(name) is not None, "imported": False}
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            dependencies[name] = {"found": False, "imported": False, "error": str(exc)}
    model_result = inspect_model(model)
    detector = inspect_detector(repo_root)
    prerequisites = recommended and all(entry["found"] for entry in dependencies.values())
    prerequisites = prerequisites and model_result["fileVerified"] and detector["inferenceImplementationFound"]
    return {"python": {"version": list(version), "recommendedRange": "3.10-3.12",
                       "recommended": recommended}, "dependencies": dependencies,
            "model": model_result, "detector": detector,
            "staticPrerequisitesSatisfied": bool(prerequisites),
            "environmentReady": bool(prerequisites),
            "runtimeReady": False, "modelLoaded": False, "inferenceValidated": False,
            "precisionTestPerformed": False,
            "notice": "Static checks only. Exit 0 means prerequisites allow a separate runtime validation, not that the model works."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local ONNX/PT file; never loaded or executed")
    parser.add_argument("--report", help="Optional new JSON report path; existing files are preserved")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Repository root containing Desktop/models/detector.py")
    args = parser.parse_args(argv)
    report = verify(args.repo_root, args.model)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        try:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x", encoding="utf-8") as output:
                output.write(text + "\n")
        except OSError as exc:
            print("Could not save new report: " + str(exc), file=sys.stderr)
            return 2
    return 0 if report["staticPrerequisitesSatisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

