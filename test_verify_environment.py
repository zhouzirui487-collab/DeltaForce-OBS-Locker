import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import verify_environment as checker


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.detector = self.root / "Desktop/models/detector.py"
        self.detector.parent.mkdir(parents=True)
        self.detector.write_text("class Detector:\n def predict(self,x):\n  return self.model.run(None, {'images': x})\n")
        self.model = self.root / "test.onnx"
        self.model.write_bytes(b"fixture only; not a real ONNX")

    def verify(self, **kwargs):
        return checker.verify(self.root, str(self.model), finder=lambda _: object(), version=(3, 11, 0), **kwargs)

    def test_file_verification_is_not_model_loading(self):
        report = self.verify()
        self.assertTrue(report["model"]["fileVerified"])
        self.assertEqual(len(report["model"]["sha256"]), 64)
        self.assertTrue(report["staticPrerequisitesSatisfied"])
        self.assertFalse(report["runtimeReady"])
        self.assertFalse(report["modelLoaded"])
        self.assertFalse(report["model"]["formatValidated"])

    def test_missing_model(self):
        self.model.unlink()
        self.assertFalse(self.verify()["staticPrerequisitesSatisfied"])

    def test_no_model(self):
        self.assertFalse(checker.inspect_model(None)["fileVerified"])

    def test_empty_and_executable_rejected(self):
        self.model.write_bytes(b"")
        self.assertFalse(checker.inspect_model(str(self.model))["fileVerified"])
        exe = self.root / "model.exe"; exe.write_bytes(b"MZ")
        self.assertFalse(checker.inspect_model(str(exe))["fileVerified"])

    def test_random_placeholder(self):
        self.detector.write_text("class Detector:\n def predict(self,x):\n  return np.random.randn(1,10,6)\n")
        result = self.verify()
        self.assertTrue(result["detector"]["placeholder"])
        self.assertFalse(result["staticPrerequisitesSatisfied"])

    def test_missing_source(self):
        self.detector.unlink()
        self.assertFalse(self.verify()["staticPrerequisitesSatisfied"])

    def test_missing_dependency(self):
        result = checker.verify(self.root, str(self.model), finder=lambda _: None, version=(3, 11, 0))
        self.assertFalse(result["staticPrerequisitesSatisfied"])
        self.assertFalse(any(item["imported"] for item in result["dependencies"].values()))

    def test_python_outside_recommendation(self):
        result = checker.verify(self.root, str(self.model), finder=lambda _: object(), version=(3, 9, 0))
        self.assertFalse(result["staticPrerequisitesSatisfied"])

    def test_existing_report_preserved(self):
        report = self.root / "report.json"; report.write_text("keep")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = checker.main(["--repo-root", str(self.root), "--report", str(report)])
        self.assertEqual(code, 2)
        self.assertEqual(report.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

