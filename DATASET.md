# Dataset Guide

## 命名

每张图片使用 `<label>_<unique-id>.<ext>`，例如 `aB19x_000042.png`。第一个下划线
之前为标签；标签只允许 `0-9a-zA-Z`，长度不超过 10。不要把含个人信息、密钥或无授权
来源的图片提交到仓库。

## 划分

将原始图片置于仓库外，再执行：

```bash
python -m src.data_preparation /path/to/raw-images --output data --seed 42
```

脚本固定按文件列表随机打乱后做 8:1:1 的 train/val/test 划分。避免同源、同模板或同一
生成批次跨集合造成数据泄漏。公开指标应记录数据版本、随机种子、样本数和评估脚本。

本仓库只保留 `data/sample/README.md`，不分发原始或派生数据集。
