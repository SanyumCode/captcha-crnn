# Minimal CRNN (legacy)

这里原样保留 `CapchaSolver` 的四个最小实现源码以便追溯。它使用灰度可变宽输入、CSV
标签和 blank=0，与 `src/` 主实现的 RGB 固定尺寸及 blank=62 不兼容。该目录不是推荐
入口，可能需要从本目录运行并自行准备 `dataset/`。
