# 本地环境只读检查

推荐使用 Python 3.10–3.12。在仓库根目录执行：

检查脚本本身仅依赖 Python 标准库，不需要安装插件。若要为后续推理准备独立环境，可在 Windows PowerShell 执行以下命令；这些命令不会启动上游程序：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy opencv-python PyYAML onnxruntime PyQt6 torch
.\.venv\Scripts\python.exe verify_environment.py --report preflight.json
```

这里的 ONNX Runtime 为 CPU 环境。GPU 后端与模型格式须在获得真实权重后单独测试，不能用安装成功证明 NVIDIA/AMD 推理可用。依赖来自 PyPI；安装时产生的版本应记录在本地 `pip freeze` 报告中。

```text
python verify_environment.py --model "本地模型.onnx" --report "新检查报告.json"
python -m unittest -v test_verify_environment.py
```

`--model`、`--report` 均可省略；缺少模型会明确返回未满足条件。若脚本不在仓库根目录，用 `--repo-root` 指向包含 `Desktop/models/detector.py` 的目录。已有报告不会被覆盖。

检查通过 `find_spec` 确认 numpy、cv2、yaml、onnxruntime、PyQt6、torch 是否可被发现，不导入这些包。对 ONNX/PT 文件只核验存在、非空和 SHA256，不反序列化、不加载、不执行。对 `predict()` 进行静态 AST 审查，发现 `np.random` 或占位实现会明确标为不可用。

退出码 0 只表示推荐 Python、依赖、非空模型文件和静态推理调用条件齐全，可开始另行验证；退出码 2 表示缺项或检查失败。`fileVerified=true` 不等于模型有效或来源可靠，`runtimeReady`、`modelLoaded` 始终保持 false，因为本脚本没有实际推理。它不是精度、实战性能或 GPU 兼容性测试。

截至 2026-10-08，上游 YOLO-omni README 仍将预训练权重列为 IN PROGRESS，公开发布中没有找到可下载权重。当前 DeltaForce-OBS-Locker 的 `predict()` 返回随机数组，安装依赖或提供某个模型文件不会自动修复该问题。不能把已有 Delta v4 冒称 YOLO-omni。

本工具不自动安装未知环境，不下载模型或二进制，不启动 APK/安装器，不进行隐藏、反检测、自动瞄准或游戏输入。

