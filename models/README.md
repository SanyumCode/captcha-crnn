# Models

模型二进制不纳入 Git，通过
[v1.0.0 Release](https://github.com/SanyumCode/captcha-crnn/releases/tag/v1.0.0)
分发：

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `best_model.pth` | PyTorch 训练 checkpoint | `b1a421c3a39f4f4b92abbd018fdccff104bd8bb55e1248a7c80e3a1110e6de52` |
| `model.pt` | TorchScript CPU 推理 | `35149a07fc49b694279fedc5cf12da51be6ad6e57938a980164fdb75fcc5547c` |
| `model.onnx` | ONNX Runtime 跨语言推理 | `7d6d6867986056374640b92709f0cace0b784631448649f2f9eba83458bbf4e2` |

仅从可信 Release 下载并核对校验值。模型评估依赖原始数据版本，训练数据不随仓库分发。
