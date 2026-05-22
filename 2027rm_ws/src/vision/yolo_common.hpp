#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo_app.hpp"

enum class TargetColor
{
    Any,
    Red,
    Blue,
};

struct Detection
{
    cv::Rect box;
    int class_id = -1;
    float confidence = 0.0f;
    bool has_keypoints = false;
    std::array<cv::Point2f, 4> keypoints = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f)};
};

struct LetterBoxInfo
{
    float scale = 1.0f;
    int pad_w = 0;
    int pad_h = 0;
};

std::string ToLowerCopy(std::string s);
TargetColor ParseTargetColor(const std::string &color);
bool IsFoxgloveDebugEnabled();
bool IsE2eLogEnabled();
bool IsYoloStopRequested();
void RequestYoloStop();
void ClearYoloMessages();
void DumpFrameForFoxglove(const cv::Mat &img);
double ElapsedMsSince(const std::chrono::steady_clock::time_point &start_tp);
std::array<cv::Point2f, 4> OrderCorners(const std::array<cv::Point2f, 4> &points);
void DrawArmorFrame(cv::Mat &vis,
                    const std::array<cv::Point2f, 4> &corners,
                    const cv::Scalar &accent_color);
std::string ResolveModelPath(const std::string &model_path);
std::vector<std::string> LoadModelLabels(const std::string &model_path);
