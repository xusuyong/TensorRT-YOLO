English | [简体中文](README.md)

# Semantic Segmentation C++ Example

This example demonstrates how to perform high-performance inference and visualization of YOLO26 and Ultralytics semantic segmentation TensorRT engines using `TensorRT-YOLO`.

## 1. Build Example

Ensure `TensorRT-YOLO` has been built and installed (e.g. to `/home/xsy/cppcode/TensorRT-YOLO/tensorrt-yolo_install_dir`):

```bash
cd examples/semantic
cmake -S . -B build -D TensorRT-YOLO_DIR=/home/xsy/cppcode/TensorRT-YOLO/tensorrt-yolo_install_dir
cmake --build build -j$(nproc)
```

After building, the executable will be at `bin/semantic`.

## 2. Run Inference

### Single Image Inference and Visualization

```bash
./bin/semantic -e /path/to/semantic.engine -i /path/to/image.jpg -l ./labels.txt -o ./output
```

### Batch Directory Inference with Profiling

```bash
./bin/semantic -e /path/to/semantic.engine -i /path/to/images_dir -l ./labels.txt -o ./output
```

## 3. Arguments

- `-e, --engine`: Path to the serialized TensorRT engine (required).
- `-i, --input`: Path to an image file or directory containing images (required).
- `-o, --output`: Directory where to save visualization results (optional).
- `-l, --labels`: Path to class labels file (default: `./labels.txt`).
