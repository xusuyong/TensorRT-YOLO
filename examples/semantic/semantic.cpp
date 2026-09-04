/**
 * @file semantic.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief Semantic Segmentation C++ 示例
 * @date 2026-09-02
 *
 * @copyright Copyright (c) 2026 laugh12321. All Rights Reserved.
 *
 */
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "trtyolo.hpp"

namespace fs = std::filesystem;

// 获取指定目录中的图像文件
std::vector<std::string> get_images_in_directory(const std::string& folder_path) {
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        const auto extension = entry.path().extension().string();
        if (fs::is_regular_file(entry) && (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp")) {
            image_files.push_back(entry.path().string());
        }
    }
    return image_files;
}

// 创建输出目录
void create_output_directory(const std::string& output_path) {
    if (!fs::exists(output_path) && !fs::create_directories(output_path)) {
        throw std::runtime_error("Failed to create output directory: " + output_path);
    } else if (!fs::is_directory(output_path)) {
        throw std::runtime_error("Output path exists but is not a directory: " + output_path);
    }
}

// 从文件中生成标签
std::vector<std::string> generate_labels(const std::string& label_file) {
    std::ifstream file(label_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + label_file);
    }

    std::vector<std::string> labels;
    std::string              label;
    while (std::getline(file, label)) {
        if (!label.empty() && label.back() == '\r') {
            label.pop_back();
        }
        labels.emplace_back(label);
    }
    return labels;
}

// 类别颜色生成
cv::Scalar get_class_color(int cls) {
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 0, 0),        // 0: 背景
        cv::Scalar(255, 56, 56),    // 1: 亮红
        cv::Scalar(56, 255, 56),    // 2: 亮绿
        cv::Scalar(56, 56, 255),    // 3: 亮蓝
        cv::Scalar(255, 157, 0),    // 4: 橙黄
        cv::Scalar(255, 112, 31),   // 5: 珊瑚
        cv::Scalar(255, 255, 0),    // 6: 黄
        cv::Scalar(255, 0, 255),    // 7: 品红
        cv::Scalar(0, 255, 255),    // 8: 青
        cv::Scalar(128, 0, 128),    // 9: 紫
        cv::Scalar(0, 128, 128),    // 10: 蓝绿
        cv::Scalar(128, 128, 0),    // 11: 橄榄
        cv::Scalar(0, 165, 255),    // 12: 深橙
        cv::Scalar(203, 192, 255),  // 13: 粉红
        cv::Scalar(144, 238, 144),  // 14: 浅绿
        cv::Scalar(230, 216, 173)   // 15: 浅蓝
    };

    if (cls >= 0 && cls < static_cast<int>(palette.size())) {
        return palette[cls];
    }

    // 超出预设范围时基于 HSV 生成可区分色彩
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((cls * 37) % 180, 210, 240));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const auto& pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

// 在图像上可视化语义分割推理结果
void visualize(cv::Mat& image, const trtyolo::SemanticRes& result, const std::vector<std::string>& labels) {
    if (result.width != image.cols || result.height != image.rows || result.mask.empty()) {
        return;
    }

    bool skip_zero = (labels.empty() || labels[0] == "background" || labels[0] == "bg");
    std::vector<bool> detected_classes(256, false);

    // 半透明融合分割掩码
    const float alpha = 0.5f;
    const float beta  = 1.0f - alpha;

    for (int y = 0; y < image.rows; ++y) {
        const uint8_t* mask_row = result.mask.data() + y * image.cols;
        cv::Vec3b*     img_row  = image.ptr<cv::Vec3b>(y);

        for (int x = 0; x < image.cols; ++x) {
            uint8_t cls = mask_row[x];
            if (skip_zero && cls == 0) {
                continue;
            }

            detected_classes[cls] = true;
            cv::Scalar col        = get_class_color(cls);

            img_row[x][0] = static_cast<uint8_t>(img_row[x][0] * beta + col[0] * alpha);
            img_row[x][1] = static_cast<uint8_t>(img_row[x][1] * beta + col[1] * alpha);
            img_row[x][2] = static_cast<uint8_t>(img_row[x][2] * beta + col[2] * alpha);
        }
    }

    // 绘制图例 (Legend)
    int legend_x = 15;
    int legend_y = 25;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.55;
    int thickness = 1;

    for (int c = 0; c < 256; ++c) {
        if (!detected_classes[c]) {
            continue;
        }

        std::string class_name = (c < static_cast<int>(labels.size())) ? labels[c] : ("class_" + std::to_string(c));
        cv::Scalar color = get_class_color(c);

        int base_line = 0;
        cv::Size text_size = cv::getTextSize(class_name, font_face, font_scale, thickness, &base_line);

        // 绘制图例色块与标签背景
        cv::Rect box_rect(legend_x, legend_y - text_size.height - 2, text_size.width + 30, text_size.height + 8);
        if (box_rect.y >= 0 && box_rect.y + box_rect.height <= image.rows &&
            box_rect.x + box_rect.width <= image.cols) {
            cv::rectangle(image, box_rect, cv::Scalar(40, 40, 40), -1);
            cv::rectangle(image, cv::Rect(legend_x + 3, legend_y - text_size.height + 1, 14, 14), color, -1);
            cv::putText(image, class_name, cv::Point(legend_x + 23, legend_y), font_face, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            legend_y += text_size.height + 14;
        }
    }
}

// 解析命令行参数
void parse_arguments(int argc, char** argv, std::string& engine_path, std::string& input_path, std::string& output_path, std::string& label_path) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " -e <engine> -i <input> [-o <output>] [-l <labels>]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-e" || arg == "--engine") {
            engine_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            input_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            output_path = argv[++i];
        } else if (arg == "-l" || arg == "--labels") {
            label_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

// 处理单张图像
void process_single_image(const std::string& image_path, const std::string& output_path, trtyolo::SemanticModel& model, const std::vector<std::string>& labels) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image from path: " + image_path);
    }

    trtyolo::Image img(image.data, image.cols, image.rows);
    auto           result = model.predict(img);

    std::cout << "Predicted image: " << image_path
              << " (" << result.width << "x" << result.height << ")"
              << ", detected classes count: " << result.num << " [";
    for (size_t i = 0; i < result.classes.size(); ++i) {
        int c = result.classes[i];
        std::string name = (c < static_cast<int>(labels.size())) ? labels[c] : std::to_string(c);
        std::cout << name << (i + 1 < result.classes.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;

    if (!output_path.empty()) {
        visualize(image, result, labels);
        fs::path output_file_path = output_path / fs::path(image_path).filename();
        cv::imwrite(output_file_path.string(), image);
        std::cout << "Saved visualization to: " << output_file_path.string() << std::endl;
    }
}

// 处理一批图像
void process_batch_images(const std::vector<std::string>& image_paths, const std::string& output_path, trtyolo::SemanticModel& model, const std::vector<std::string>& labels) {
    const int batch_size = model.batch();
    for (size_t i = 0; i < image_paths.size(); i += batch_size) {
        std::vector<cv::Mat>        images;
        std::vector<trtyolo::Image> img_batch;
        std::vector<std::string>    img_name_batch;

        for (size_t j = i; j < i + batch_size && j < image_paths.size(); ++j) {
            cv::Mat image = cv::imread(image_paths[j], cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("Failed to read image from path: " + image_paths[j]);
            }
            images.push_back(image);
            img_batch.emplace_back(image.data, image.cols, image.rows);
            img_name_batch.push_back(fs::path(image_paths[j]).filename().string());
        }

        auto results = model.predict(img_batch);

        if (!output_path.empty()) {
            for (size_t j = 0; j < images.size(); ++j) {
                visualize(images[j], results[j], labels);
                fs::path output_file_path = output_path + "/" + img_name_batch[j];
                cv::imwrite(output_file_path.string(), images[j]);
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        std::string engine_path, input_path, output_path, label_path;
        parse_arguments(argc, argv, engine_path, input_path, output_path, label_path);

        if (!fs::exists(engine_path)) {
            throw std::runtime_error("Engine path does not exist: " + engine_path);
        }
        if (!fs::exists(input_path) || (!fs::is_regular_file(input_path) && !fs::is_directory(input_path))) {
            throw std::runtime_error("Input path does not exist or is not a regular file/directory: " + input_path);
        }

        std::vector<std::string> labels;
        if (!output_path.empty()) {
            if (label_path.empty()) {
                throw std::runtime_error("Please provide a labels file using -l or --labels.");
            }
            if (!fs::exists(label_path)) {
                throw std::runtime_error("Label path does not exist: " + label_path);
            }
            labels = generate_labels(label_path);
            create_output_directory(output_path);
        }

        trtyolo::InferOption option;
        option.enableSwapRB();

        if (!fs::is_regular_file(input_path)) {
            option.enablePerformanceReport();
        }

        auto model = std::make_unique<trtyolo::SemanticModel>(engine_path, option);

        if (fs::is_regular_file(input_path)) {
            process_single_image(input_path, output_path, *model, labels);
        } else {
            auto image_files = get_images_in_directory(input_path);
            if (image_files.empty()) {
                throw std::runtime_error("Failed to read image from path: " + input_path);
            }
            process_batch_images(image_files, output_path, *model, labels);
        }

        std::cout << "Inference completed." << std::endl;

        if (!fs::is_regular_file(input_path)) {
            auto [throughput_str, gpu_latency_str, cpu_latency_str] = model->performanceReport();
            std::cout << throughput_str << std::endl;
            std::cout << gpu_latency_str << std::endl;
            std::cout << cpu_latency_str << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
