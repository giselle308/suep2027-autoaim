#pragma once

#include <functional>
#include <vector>

#include <openvino/openvino.hpp>

#include "yolo_common.hpp"

class YoloPostprocessor
{
public:
    YoloPostprocessor(int input_w, int input_h, int num_classes, float conf_thres, float nms_thres);

    std::vector<Detection> run(const cv::Mat &orig,
                               ov::Tensor &out,
                               const LetterBoxInfo &lb,
                               const AppConfig &cfg,
                               const std::function<bool(int)> &class_filter = {}) const;

private:
    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 80;
    float conf_thres_ = 0.25f;
    float nms_thres_ = 0.45f;
};
