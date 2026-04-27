#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "postprocess.hpp"
#include "yolo_app.hpp"

class YoloOpenvino
{
private:
    struct AsyncSlot
    {
        ov::InferRequest request;
        bool busy = false;
        int infer_id = 0;
        cv::Mat frame;
        cv::Mat resized_img;
        cv::Mat blob;
        uint64_t frame_id = 0;
        std::chrono::steady_clock::time_point capture_tp;
        LetterBoxInfo lb;
    };

public:
    bool init(const AppConfig &cfg, std::string *error);
    bool hasIdleRequest() const;
    bool submit(const FrameMParam &frame_msg, std::string *error);
    bool collectCompleted(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error);
    bool waitAll(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error);

private:
    bool finishSlot(AsyncSlot &slot, std::shared_ptr<ResultMParam> &result, std::string *error);
    bool MatchTargetColorByClass(int class_id, TargetColor target) const;
    void preprocess(const cv::Mat &bgr, cv::Mat &resized_img, cv::Mat &blob, LetterBoxInfo &lb) const;

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
    std::vector<std::string> class_labels_;
    std::optional<YoloPostprocessor> postprocessor_;
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
