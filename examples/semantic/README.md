[English](README.en.md) | 简体中文

# 语义分割 (Semantic Segmentation) C++ 示例

本示例展示如何使用 `TensorRT-YOLO` 运行 YOLO26 及 Ultralytics 语义分割（Semantic Segmentation）TensorRT 引擎进行高效推理与可视化。

## 1. 编译示例

请确保已编译并安装主库 `TensorRT-YOLO` 到指定目录（例如 `/home/xsy/cppcode/TensorRT-YOLO/tensorrt-yolo_install_dir`）：

```bash
cd examples/semantic
cmake -S . -B build -D TensorRT-YOLO_DIR=/home/xsy/cppcode/TensorRT-YOLO/tensorrt-yolo_install_dir
cmake --build build -j$(nproc)
```

编译完成后，可执行程序位于 `bin/semantic`。

## 2. 运行推理

### 单张图片推理并可视化保存

```bash
./bin/semantic -e /path/to/semantic.engine -i /path/to/image.jpg -l ./labels.txt -o ./output
```

### 批量图像推理与性能统计

```bash
./bin/semantic -e /path/to/semantic.engine -i /path/to/images_dir -l ./labels.txt -o ./output
```

## 3. 参数说明

- `-e, --engine`: 序列化的 TensorRT 引擎路径（必需）。
- `-i, --input`: 输入单张图片路径或图片文件夹路径（必需）。
- `-o, --output`: 可视化保存结果目录（可选）。
- `-l, --labels`: 类别标签文件路径（默认为 `./labels.txt`）。
