#pragma once

#include <array>
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
                               const std::function<bool(int)> &class_filter = {});

private:
    struct OutputLayoutCache
    {
        std::vector<std::size_t> shape;
        bool channel_first = true;
        int channels = 0;
        int points = 0;
        int use_classes = 0;
        bool expected_anchor_points = false;
        bool valid = false;
    };

    struct Scratch
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> scores;
        std::vector<bool> has_keypoints;
        std::vector<std::array<cv::Point2f, 4>> keypoints;
        std::vector<int> candidate_indices;
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> local_keep;
        std::vector<Detection> detections;
    };

    const OutputLayoutCache &resolveOutputLayout(const ov::Shape &shape, const float *data);

    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 80;
    float conf_thres_ = 0.25f;
    float nms_thres_ = 0.45f;
    OutputLayoutCache output_layout_;
    Scratch scratch_;
};
