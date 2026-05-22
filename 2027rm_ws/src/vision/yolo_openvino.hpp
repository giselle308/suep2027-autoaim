#pragma once

#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "message_pool.hpp"
#include "memory_layout.hpp"
#include "postprocess.hpp"
#include "yolo_app.hpp"

class YoloOpenvino
{
private:
    struct alignas(app::memory::kCacheLineSize) AsyncSlot
    {
        ov::InferRequest request;
        bool busy = false;
        int infer_id = 0;
        cv::Mat frame;
        cv::Mat resized_img;
        cv::Mat letterbox_img;
        uint64_t frame_id = 0;
        std::chrono::steady_clock::time_point pipeline_start_tp;
        std::chrono::steady_clock::time_point capture_tp;
        std::chrono::steady_clock::time_point submit_tp;
        LetterBoxInfo lb;
        cv::Size cached_src_size;
        int cached_resize_w = 0;
        int cached_resize_h = 0;
        bool letterbox_cache_valid = false;
        bool letterbox_padding_valid = false;
    };

    struct alignas(app::memory::kCacheLineSize) CompletionState
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool notified = false;
        std::exception_ptr exception;
    };

public:
    bool init(const AppConfig &cfg, std::string *error);
    bool hasIdleRequest() const;
    bool submit(const FrameMParam &frame_msg, std::string *error);
    bool collectCompleted(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error);
    bool waitAll(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error);
    void waitForCompletionSignal(std::chrono::milliseconds timeout);

private:
    bool finishSlot(AsyncSlot &slot, std::shared_ptr<ResultMParam> &result, std::string *error);
    void rebuildAllowedClassTable();
    void preprocess(AsyncSlot &slot) const;
    void notifyRequestCompleted(std::exception_ptr exception);
    bool consumeCallbackException(std::string *error);

    ov::Core core_;
    ov::CompiledModel compiled;
    ov::Output<const ov::Node> input_port_;
    ov::Output<const ov::Node> output_port_;
    std::vector<AsyncSlot> slots_;
    int async_request_count_ = 1;
    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 80;
    float conf_thres_ = 0.25f;
    float nms_thres_ = 0.45f;
    TargetColor target_color_ = TargetColor::Any;
    int max_color_candidates_ = 1;
    std::vector<std::string> class_labels_;
    std::vector<uint8_t> class_allowed_;
    std::optional<YoloPostprocessor> postprocessor_;
    SharedParamPool<ResultMParam> result_pool_;
    CompletionState completion_;
};

class YoloResultFilter
{
public:
    void reset();
    void apply(ResultMParam &result, const AppConfig &cfg);

private:
    class EmaScalar
    {
    public:
        void reset();
        double filter(double x, double alpha);

    private:
        bool initialized_ = false;
        double prev_ = 0.0;
    };

    bool initialized_ = false;
    uint64_t last_frame_id_ = 0;
    std::array<EmaScalar, 8> filters_;
};
