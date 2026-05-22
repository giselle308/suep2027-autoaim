#pragma once

#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>

class HikCameraNode {
public:
    HikCameraNode() = default;
    ~HikCameraNode();

    bool init(std::string *error = nullptr);
    bool grab(cv::Mat &bgr,
              std::chrono::steady_clock::time_point *capture_tp = nullptr,
              std::string *error = nullptr);
    void shutdown();
    int outputWidth() const { return output_width_; }
    int outputHeight() const { return output_height_; }

private:
    void *handle_ = nullptr;
    int output_width_ = 0;
    int output_height_ = 0;
    bool alignment_logged_ = false;
};
