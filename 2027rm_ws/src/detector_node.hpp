#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class RedLightBarDetector {
public:
    RedLightBarDetector() = default;

    bool init(float depth_cm,
              const std::string &camera_info_yaml,
              float expected_len_cm = 6.0f,
              float expected_gap_cm = 14.0f,
              std::string *error = nullptr);

    bool detect(const cv::Mat &frame,
                cv::Mat &vis,
                bool *found = nullptr,
                std::string *error = nullptr) const;

private:
    float expected_len_px_ = 0.0f;
    float expected_gap_px_ = 0.0f;
    float len_tol_ratio_ = 0.35f;
    float gap_tol_ratio_ = 0.35f;
    float parallel_deg_th_ = 10.0f;
    bool initialized_ = false;
};
