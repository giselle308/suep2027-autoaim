#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class HikCameraNode {
public:
    HikCameraNode() = default;
    ~HikCameraNode();

    bool init(std::string *error = nullptr);
    bool grab(cv::Mat &bgr, std::string *error = nullptr);
    void shutdown();

private:
    void *handle_ = nullptr;
    int bayer_cvt_code_ = cv::COLOR_BayerRG2BGR;
};
