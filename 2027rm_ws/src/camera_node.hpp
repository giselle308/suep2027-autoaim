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
              double *grab_ms = nullptr,
              double *pixel_convert_ms = nullptr,
              std::string *error = nullptr);
    void shutdown();

private:
    void *handle_ = nullptr;
    bool alignment_logged_ = false;
};
