/**
 * @file trtyolo.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT YOLO 模型推理相关类和结构体的实现
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector_functions.hpp>

#include "backend.hpp"
#include "utils/common.hpp"

namespace trtyolo {

Image::Image(void* data, int width, int height) : ptr(data), width(width), height(height), channels(3), pitch(width * sizeof(uint8_t) * 3) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
}

Image::Image(void* data, int width, int height, size_t pitch)
    : ptr(data), width(width), height(height), channels(3), pitch(pitch) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width, height and channels must be positive"));
    }
    if (pitch < static_cast<size_t>(width * sizeof(uint8_t) * channels)) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: pitch must >= width * channels"));
    }
}

Image::Image(void* data, int width, int height, int channels, size_t pitch)
    : ptr(data), width(width), height(height), channels(channels), pitch(pitch) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width, height and channels must be positive"));
    }
    if (pitch < static_cast<size_t>(width * sizeof(uint8_t) * channels)) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: pitch must >= width * channels"));
    }
}

Mask::Mask(int width, int height) : width(width), height(height) {
    if (width < 0 || height < 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Mask: width and height must be positive"));
    }
    data.resize(width * height);
}

std::array<int, 4> Box::xyxy() const {
    return {static_cast<int>(left), static_cast<int>(top), static_cast<int>(right), static_cast<int>(bottom)};
}

std::array<int, 8> RotatedBox::xyxyxyxy() const {
    float cx   = (left + right) * 0.5f;
    float cy   = (top + bottom) * 0.5f;
    float dx   = (right - left) * 0.5f;
    float dy   = (bottom - top) * 0.5f;
    float cosa = std::cos(theta);
    float sina = std::sin(theta);

    auto rot = [&](float px, float py) -> std::pair<int, int> {
        int x = static_cast<int>(std::round(px * cosa - py * sina + cx));
        int y = static_cast<int>(std::round(px * sina + py * cosa + cy));
        return {x, y};
    };

    auto [x1, y1] = rot(-dx, -dy);  // 左上
    auto [x2, y2] = rot(dx, -dy);   // 右上
    auto [x3, y3] = rot(dx, dy);    // 右下
    auto [x4, y4] = rot(-dx, dy);   // 左下

    return {x1, y1, x2, y2, x3, y3, x4, y4};
}

class InferOption::Impl {
public:
    InferConfig getInferConfig() const { return infer_config; }
    void        setDeviceId(int id) { infer_config.device_id = id; }
    void        enableCudaMem() { infer_config.cuda_mem = true; }
    void        enableManagedMemory() { infer_config.enable_managed_memory = true; }
    void        enablePerformanceReport() { infer_config.enable_performance_report = true; }
    void        setInputDimensions(int width, int height) { infer_config.input_shape = make_int2(height, width); }
    void        enableSwapRB() { infer_config.config.swap_rb = true; }
    void        setBorderValue(float value) { infer_config.config.border_value = value; }
    void        setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) {
        assert(mean.size() == 3 && std.size() == 3 && "ProcessConfig: requires the size of mean and std to be 3.");

        infer_config.config.alpha.x = 1.0 / 255.0f / std[0];
        infer_config.config.alpha.y = 1.0 / 255.0f / std[1];
        infer_config.config.alpha.z = 1.0 / 255.0f / std[2];
        infer_config.config.beta.x  = -mean[0] / std[0];
        infer_config.config.beta.y  = -mean[1] / std[1];
        infer_config.config.beta.z  = -mean[2] / std[2];
    }

private:
    InferConfig infer_config;  // < 推理选项配置
};

InferOption::InferOption() : impl_(std::make_unique<InferOption::Impl>()) {}
InferOption::~InferOption() = default;

void InferOption::setDeviceId(int id) { impl_->setDeviceId(id); }
void InferOption::enableCudaMem() { impl_->enableCudaMem(); }
void InferOption::enableManagedMemory() { impl_->enableManagedMemory(); }
void InferOption::enablePerformanceReport() { impl_->enablePerformanceReport(); }
void InferOption::enableSwapRB() { impl_->enableSwapRB(); }
void InferOption::setBorderValue(float border_value) { impl_->setBorderValue(border_value); }
void InferOption::setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) { impl_->setNormalizeParams(mean, std); }
void InferOption::setInputDimensions(int width, int height) { impl_->setInputDimensions(height, width); }

class BaseModel::Impl {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    Impl()  = default;
    ~Impl() = default;

    Impl(const std::string& trt_engine_file, const InferOption& infer_option)
        : backend_(std::make_unique<TrtBackend>(trt_engine_file, infer_option.impl_->getInferConfig())) {
        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_ = std::make_unique<GpuTimer>(backend_->stream);
            infer_cpu_trace_ = std::make_unique<CpuTimer>();
        }
    }

    std::unique_ptr<Impl> clone() const {
        auto clone_impl              = std::make_unique<Impl>();
        clone_impl->backend_         = backend_->clone();
        clone_impl->infer_gpu_trace_ = std::make_unique<GpuTimer>(clone_impl->backend_->stream);
        clone_impl->infer_cpu_trace_ = std::make_unique<CpuTimer>();
        return clone_impl;
    }

    std::tuple<std::string, std::string, std::string> performanceReport() {
        if (backend_->infer_config.enable_performance_report) {
            float             throughput = total_request_ / infer_cpu_trace_->totalMilliseconds() * 1000;
            std::stringstream ss;

            // 构建吞吐量字符串
            ss << "Throughput: " << throughput << " qps";
            std::string throughputStr = ss.str();
            ss.str("");  // 清空 stringstream

            auto percentiles = std::vector<float>{90, 95, 99};

            auto getLatencyStr = [&](const auto& trace, const std::string& device) {
                auto result = getPerformanceResult(trace->milliseconds(), {0.90, 0.95, 0.99});
                ss << device << " Latency: min = " << result.min << " ms, max = " << result.max
                   << " ms, mean = " << result.mean << " ms, median = " << result.median << " ms";
                for (int32_t i = 0, n = percentiles.size(); i < n; ++i) {
                    ss << ", percentile(" << percentiles[i] << "%) = " << result.percentiles[i] << " ms";
                }
                std::string output = ss.str();
                ss.str("");  // 清空 stringstream
                return output;
            };

            std::string cpuLatencyStr = getLatencyStr(infer_cpu_trace_, "CPU");
            std::string gpuLatencyStr = getLatencyStr(infer_gpu_trace_, "GPU");

            total_request_ = 0;
            infer_cpu_trace_->reset();
            infer_gpu_trace_->reset();

            return std::make_tuple(throughputStr, cpuLatencyStr, gpuLatencyStr);
        }
        // 性能报告未启用时返回空字符串
        return std::make_tuple("", "", "");
    }

    size_t batch() const {
        return backend_->max_shape.x;
    }

    // 装饰器函数
    template <typename Func, typename ReturnType>
    ReturnType withPerformanceReport(const std::vector<Image>& images, Func func) {
        if (backend_->infer_config.enable_performance_report) {
            total_request_ += (backend_->dynamic ? images.size() : backend_->max_shape.x);
            infer_cpu_trace_->start();
            infer_gpu_trace_->start();
        }

        backend_->infer(images);  // 调用推理方法
        ReturnType result = func(images.size());

        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_->stop();
            infer_cpu_trace_->stop();
        }

        return result;
    }

    // ClassifyModel 的后处理方法实现
    ClassifyRes postProcessClassify(int idx) {
        auto&  tensor_info = backend_->tensor_infos[1];
        float* topk        = static_cast<float*>(tensor_info.buffer->host()) + idx * tensor_info.shape.d[1] * tensor_info.shape.d[2];

        ClassifyRes result;
        result.num = tensor_info.shape.d[1];
        result.scores.reserve(result.num);
        result.classes.reserve(result.num);

        for (int i = 0; i < result.num; ++i) {
            result.scores.push_back(topk[i * tensor_info.shape.d[2]]);
            result.classes.push_back(topk[i * tensor_info.shape.d[2] + 1]);
        }

        return result;
    }

    // DetectModel 的后处理方法实现
    DetectRes postProcessDetect(int idx) {
        auto& num_tensor   = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& class_tensor = backend_->tensor_infos[4];

        int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

        DetectRes result;
        result.num   = num;
        int box_size = box_tensor.shape.d[2];

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(num);
        result.scores.reserve(num);
        result.classes.reserve(num);

        for (int i = 0; i < num; ++i) {
            int   base_index = i * box_size;
            float left = boxes[base_index], top = boxes[base_index + 1];
            float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);
        }

        return result;
    }

    // OBBModel 的后处理方法实现
    OBBRes postProcessOBB(int idx) {
        auto& num_tensor   = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& class_tensor = backend_->tensor_infos[4];

        int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

        OBBRes result;
        result.num   = num;
        int box_size = box_tensor.shape.d[2];

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(num);
        result.scores.reserve(num);
        result.classes.reserve(num);

        for (int i = 0; i < num; ++i) {
            int   base_index = i * box_size;
            float left = boxes[base_index], top = boxes[base_index + 1];
            float right = boxes[base_index + 2], bottom = boxes[base_index + 3];
            float theta = boxes[base_index + 4];

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);
        }

        return result;
    }

    // SegmentModel 的后处理方法实现
    SegmentRes postProcessSegment(int idx) {
        auto& num_tensor   = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& class_tensor = backend_->tensor_infos[4];
        auto& mask_tensor  = backend_->tensor_infos[5];
        int   mask_height  = mask_tensor.shape.d[2];
        int   mask_width   = mask_tensor.shape.d[3];

        int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
        float* masks   = static_cast<float*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

        SegmentRes result;
        result.num   = num;
        int box_size = box_tensor.shape.d[2];

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(num);
        result.scores.reserve(num);
        result.classes.reserve(num);
        result.masks.reserve(num);

        for (int i = 0; i < num; ++i) {
            int   base_index = i * box_size;
            float left = boxes[base_index], top = boxes[base_index + 1];
            float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);

            Mask mask(mask_width, mask_height);
            // Directly copy all mask data without edge cropping
            int  start_idx = i * mask_height * mask_width;
            std::memcpy(mask.data.data(), masks + start_idx, mask_height * mask_width * sizeof(float));

            result.masks.emplace_back(std::move(mask));
        }

        return result;
    }

    // PoseModel 的后处理方法实现
    PoseRes postProcessPose(int idx) {
        auto& num_tensor   = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& class_tensor = backend_->tensor_infos[4];
        auto& kpt_tensor   = backend_->tensor_infos[5];
        int   nkpt         = kpt_tensor.shape.d[2];
        int   ndim         = kpt_tensor.shape.d[3];

        int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
        float* kpts    = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

        PoseRes result;
        result.num   = num;
        int box_size = box_tensor.shape.d[2];

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(num);
        result.scores.reserve(num);
        result.classes.reserve(num);
        result.kpts.reserve(num);

        for (int i = 0; i < num; ++i) {
            int   base_index = i * box_size;
            float left = boxes[base_index], top = boxes[base_index + 1];
            float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);

            std::vector<KeyPoint> keypoints;
            for (int j = 0; j < nkpt; ++j) {
                float x = kpts[i * nkpt * ndim + j * ndim];
                float y = kpts[i * nkpt * ndim + j * ndim + 1];
                transform.apply(x, y, &x, &y);
                keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
            }
            result.kpts.emplace_back(std::move(keypoints));
        }

        return result;
    }

    // SemanticModel 的后处理方法实现
    SemanticRes postProcessSemantic(int idx, int orig_width, int orig_height) {
        auto& tensor_info = backend_->tensor_infos[1];

        int net_h    = 0;
        int net_w    = 0;
        int channels = 1;

        if (tensor_info.shape.nbDims == 3) {
            net_h    = tensor_info.shape.d[1];
            net_w    = tensor_info.shape.d[2];
            channels = 1;
        } else if (tensor_info.shape.nbDims == 4) {
            channels = tensor_info.shape.d[1];
            net_h    = tensor_info.shape.d[2];
            net_w    = tensor_info.shape.d[3];
        } else {
            throw std::runtime_error("Unexpected output tensor dimensions for semantic segmentation: " +
                                     std::to_string(tensor_info.shape.nbDims));
        }

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        // 缩放比例以适配可能与网络输入尺寸不一致的输出特征图（通常一致）
        float scale_w = static_cast<float>(net_w) / backend_->max_shape.w;
        float scale_h = static_cast<float>(net_h) / backend_->max_shape.z;

        int cur_valid_w  = std::max(1, static_cast<int>(std::round(transform.meta.x * scale_w)));
        int cur_valid_h  = std::max(1, static_cast<int>(std::round(transform.meta.y * scale_h)));
        int cur_offset_x = static_cast<int>(std::round(transform.meta.z * scale_w));
        int cur_offset_y = static_cast<int>(std::round(transform.meta.w * scale_h));

        SemanticRes result;
        result.width  = orig_width;
        result.height = orig_height;
        result.mask.resize(orig_width * orig_height);

        // 预计算 X 轴最近邻映射表
        std::vector<int> map_x(orig_width);
        for (int x = 0; x < orig_width; ++x) {
            int sx   = cur_offset_x + static_cast<int>(x * cur_valid_w / orig_width);
            map_x[x] = std::clamp(sx, 0, net_w - 1);
        }

        bool   class_present[256]    = {false};
        size_t batch_offset_elements = static_cast<size_t>(idx) * channels * net_h * net_w;

        if (tensor_info.dtype() == nvinfer1::DataType::kUINT8) {
            const uint8_t* out_ptr = static_cast<const uint8_t*>(tensor_info.buffer->host()) + batch_offset_elements;
            for (int y = 0; y < orig_height; ++y) {
                int            sy       = cur_offset_y + static_cast<int>(y * cur_valid_h / orig_height);
                sy                      = std::clamp(sy, 0, net_h - 1);
                const uint8_t* row_src  = out_ptr + sy * net_w;
                uint8_t*       dst_row  = result.mask.data() + y * orig_width;
                for (int x = 0; x < orig_width; ++x) {
                    uint8_t cls        = row_src[map_x[x]];
                    dst_row[x]         = cls;
                    class_present[cls] = true;
                }
            }
        } else if (tensor_info.dtype() == nvinfer1::DataType::kINT32) {
            const int32_t* out_ptr = static_cast<const int32_t*>(tensor_info.buffer->host()) + batch_offset_elements;
            for (int y = 0; y < orig_height; ++y) {
                int            sy       = cur_offset_y + static_cast<int>(y * cur_valid_h / orig_height);
                sy                      = std::clamp(sy, 0, net_h - 1);
                const int32_t* row_src  = out_ptr + sy * net_w;
                uint8_t*       dst_row  = result.mask.data() + y * orig_width;
                for (int x = 0; x < orig_width; ++x) {
                    uint8_t cls        = static_cast<uint8_t>(row_src[map_x[x]]);
                    dst_row[x]         = cls;
                    class_present[cls] = true;
                }
            }
        } else if (tensor_info.dtype() == nvinfer1::DataType::kFLOAT) {
            const float* out_ptr = static_cast<const float*>(tensor_info.buffer->host()) + batch_offset_elements;
            if (channels == 1) {
                for (int y = 0; y < orig_height; ++y) {
                    int          sy      = cur_offset_y + static_cast<int>(y * cur_valid_h / orig_height);
                    sy                   = std::clamp(sy, 0, net_h - 1);
                    const float* row_src = out_ptr + sy * net_w;
                    uint8_t*     dst_row = result.mask.data() + y * orig_width;
                    for (int x = 0; x < orig_width; ++x) {
                        uint8_t cls        = static_cast<uint8_t>(std::round(row_src[map_x[x]]));
                        dst_row[x]         = cls;
                        class_present[cls] = true;
                    }
                }
            } else {
                // channels > 1: logits
                size_t plane_size = static_cast<size_t>(net_h) * net_w;
                for (int y = 0; y < orig_height; ++y) {
                    int      sy      = cur_offset_y + static_cast<int>(y * cur_valid_h / orig_height);
                    sy               = std::clamp(sy, 0, net_h - 1);
                    uint8_t* dst_row = result.mask.data() + y * orig_width;
                    for (int x = 0; x < orig_width; ++x) {
                        int    sx          = map_x[x];
                        size_t spatial_idx = sy * net_w + sx;
                        int    best_c      = 0;
                        float  max_val     = out_ptr[spatial_idx];
                        for (int c = 1; c < channels; ++c) {
                            float val = out_ptr[c * plane_size + spatial_idx];
                            if (val > max_val) {
                                max_val = val;
                                best_c  = c;
                            }
                        }
                        uint8_t cls        = static_cast<uint8_t>(best_c);
                        dst_row[x]         = cls;
                        class_present[cls] = true;
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported data type for semantic segmentation output");
        }

        for (int c = 0; c < 256; ++c) {
            if (class_present[c]) {
                result.classes.push_back(c);
            }
        }
        result.num = static_cast<int>(result.classes.size());

        return result;
    }

private:
    std::unique_ptr<TrtBackend> backend_;           // < TensorRT 后端
    unsigned long long          total_request_{0};  // < 总请求数
    std::unique_ptr<GpuTimer>   infer_gpu_trace_;   // < GPU推理计时器
    std::unique_ptr<CpuTimer>   infer_cpu_trace_;   // < CPU推理计时器
};

BaseModel::BaseModel()  = default;
BaseModel::~BaseModel() = default;

BaseModel::BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : impl_(std::make_unique<Impl>(trt_engine_file, infer_option)) {}

int BaseModel::batch() const {
    return impl_->batch();
}

std::tuple<std::string, std::string, std::string> BaseModel::performanceReport() {
    return impl_->performanceReport();
}

ClassifyModel::ClassifyModel()  = default;
ClassifyModel::~ClassifyModel() = default;

ClassifyModel::ClassifyModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<ClassifyModel> ClassifyModel::clone() const {
    auto clone_model   = std::make_unique<ClassifyModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<ClassifyRes> ClassifyModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<ClassifyRes> {
        std::vector<ClassifyRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessClassify(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<ClassifyRes>>(images, processImages);
}

ClassifyRes ClassifyModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

DetectModel::DetectModel()  = default;
DetectModel::~DetectModel() = default;

DetectModel::DetectModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<DetectModel> DetectModel::clone() const {
    auto clone_model   = std::make_unique<DetectModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<DetectRes> DetectModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<DetectRes> {
        std::vector<DetectRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessDetect(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<DetectRes>>(images, processImages);
}

DetectRes DetectModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

OBBModel::OBBModel()  = default;
OBBModel::~OBBModel() = default;

OBBModel::OBBModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<OBBModel> OBBModel::clone() const {
    auto clone_model   = std::make_unique<OBBModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<OBBRes> OBBModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<OBBRes> {
        std::vector<OBBRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessOBB(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<OBBRes>>(images, processImages);
}

OBBRes OBBModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

SegmentModel::SegmentModel()  = default;
SegmentModel::~SegmentModel() = default;

SegmentModel::SegmentModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<SegmentModel> SegmentModel::clone() const {
    auto clone_model   = std::make_unique<SegmentModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<SegmentRes> SegmentModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<SegmentRes> {
        std::vector<SegmentRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessSegment(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<SegmentRes>>(images, processImages);
}

SegmentRes SegmentModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

PoseModel::PoseModel()  = default;
PoseModel::~PoseModel() = default;

PoseModel::PoseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<PoseModel> PoseModel::clone() const {
    auto clone_model   = std::make_unique<PoseModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<PoseRes> PoseModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<PoseRes> {
        std::vector<PoseRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessPose(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<PoseRes>>(images, processImages);
}

PoseRes PoseModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

SemanticModel::SemanticModel()  = default;
SemanticModel::~SemanticModel() = default;

SemanticModel::SemanticModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<SemanticModel> SemanticModel::clone() const {
    auto clone_model   = std::make_unique<SemanticModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<SemanticRes> SemanticModel::predict(const std::vector<Image>& images) {
    auto processImages = [this, &images](size_t num) -> std::vector<SemanticRes> {
        std::vector<SemanticRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessSemantic(idx, images[idx].width, images[idx].height);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<SemanticRes>>(images, processImages);
}

SemanticRes SemanticModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

}  // namespace trtyolo
