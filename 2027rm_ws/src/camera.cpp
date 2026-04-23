#include <chrono>
#include <cctype>
#include <fstream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

#include <MvCameraControl.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "camera_node.hpp"
#include "logging.hpp"

#ifndef CAMERA_CONFIG_PATH
#define CAMERA_CONFIG_PATH "../config/config.yaml"
#endif

struct CameraConfig {
    std::string interface = "USB3Vision";
    int width = 1440;
    int height = 1080;
    std::string pixel_format = "BayerRG8";
    int frame_rate = 0;
    bool frame_rate_enable = false;
    std::string trigger_mode = "off";
    std::string trigger_source = "software";
    float exposure_time = 2000.0f;
    bool exposure_auto = false;
    float gain = 0.0f;
    bool gain_auto = false;
    std::string white_balance_mode = "manual";
    float white_balance_red = 1.0f;
    float white_balance_blue = 1.0f;
};

static std::string ToLower(std::string value) 
{
    for (char &ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

static unsigned int GetPixelFormatEnum(const std::string &fmt) 
{
    const std::string f = ToLower(fmt);
    if (f == "bayerbg8") return PixelType_Gvsp_BayerBG8;
    if (f == "bayergb8") return PixelType_Gvsp_BayerGB8;
    if (f == "bayergr8") return PixelType_Gvsp_BayerGR8;
    return PixelType_Gvsp_BayerRG8;
}

static int GetBayerCvtCode(const std::string &fmt) {
    const std::string f = ToLower(fmt);
    if (f == "bayerbg8") return cv::COLOR_BayerBG2BGR;
    if (f == "bayergb8") return cv::COLOR_BayerGB2BGR;
    if (f == "bayergr8") return cv::COLOR_BayerGR2BGR;
    return cv::COLOR_BayerRG2BGR;
}

static YAML::Node LoadConfigWithFallback(std::string &usedPath) {
    const std::vector<std::string> candidates = {
        CAMERA_CONFIG_PATH,
        "../config/config.yaml",
        "2027rm_ws/config/config.yaml",
        "config/config.yaml"
    };

    for (const auto &path : candidates) {
        std::ifstream fin(path);
        if (fin.good()) {
            usedPath = path;
            return YAML::LoadFile(path);
        }
    }

    throw std::runtime_error("bad file: config.yaml (tried absolute/relative fallback paths)");
}

static bool LoadCameraConfig(CameraConfig &cfg, std::string &usedPath, std::string *error) {
    try {
        YAML::Node config = LoadConfigWithFallback(usedPath);
        const auto camera = config["camera"];
        cfg.interface = camera["connection"]["interface"].as<std::string>();
        cfg.width = camera["image"]["width"].as<int>();
        cfg.height = camera["image"]["height"].as<int>();
        if (camera["image"]["pixel_format"]) {
            cfg.pixel_format = camera["image"]["pixel_format"].as<std::string>();
        }
        cfg.frame_rate = camera["image"]["frame_rate"].as<int>();
        cfg.frame_rate_enable = camera["image"]["frame_rate_enable"].as<bool>();
        cfg.trigger_mode = camera["trigger"]["mode"].as<std::string>();
        cfg.trigger_source = camera["trigger"]["source"].as<std::string>();
        cfg.exposure_time = camera["capture"]["exposure_time"].as<float>();
        cfg.exposure_auto = camera["capture"]["exposure_auto"].as<bool>();
        cfg.gain = camera["capture"]["gain"].as<float>();
        cfg.gain_auto = camera["capture"]["gain_auto"].as<bool>();
        cfg.white_balance_mode = camera["capture"]["white_balance_mode"].as<std::string>();
        cfg.white_balance_red = camera["capture"]["white_balance_red"].as<float>();
        cfg.white_balance_blue = camera["capture"]["white_balance_blue"].as<float>();
        return true;
    } catch (const std::exception &e) {
        if (error) {
            *error = e.what();
        }
        return false;
    }
}

HikCameraNode::~HikCameraNode() {
    shutdown();
}

bool HikCameraNode::init(std::string *error) {
    shutdown();

    CameraConfig cfg;
    std::string configPath;
    if (!LoadCameraConfig(cfg, configPath, error)) {
        return false;
    }

    spdlog::info("Loaded camera config: {} | pixel_format={} | size={}x{} | exposure={}us",
                 configPath,
                 cfg.pixel_format,
                 cfg.width,
                 cfg.height,
                 cfg.exposure_time);

    MV_CC_DEVICE_INFO_LIST deviceList{};
    int nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &deviceList);
    if (nRet != MV_OK) {
        if (error) *error = "枚举设备失败";
        return false;
    }
    if (deviceList.nDeviceNum == 0) {
        if (error) *error = "未发现相机";
        return false;
    }

    nRet = MV_CC_CreateHandle(&handle_, deviceList.pDeviceInfo[0]);
    if (nRet != MV_OK || handle_ == nullptr) {
        if (error) *error = "创建句柄失败";
        return false;
    }

    nRet = MV_CC_OpenDevice(handle_);
    if (nRet != MV_OK) {
        if (error) *error = "打开设备失败";
        shutdown();
        return false;
    }

    bayer_cvt_code_ = GetBayerCvtCode(cfg.pixel_format);
    MV_CC_SetIntValue(handle_, "Width", cfg.width);
    MV_CC_SetIntValue(handle_, "Height", cfg.height);
    MV_CC_SetEnumValue(handle_, "PixelFormat", GetPixelFormatEnum(cfg.pixel_format));

    if (cfg.trigger_mode == "off") {
        MV_CC_SetEnumValue(handle_, "TriggerMode", 0);
    } else {
        MV_CC_SetEnumValue(handle_, "TriggerMode", 1);
        if (cfg.trigger_source == "software") {
            MV_CC_SetEnumValue(handle_, "TriggerSource", 7);
        } else {
            MV_CC_SetEnumValue(handle_, "TriggerSource", 0);
        }
    }

    if (cfg.frame_rate_enable) {
        MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
        MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", static_cast<float>(cfg.frame_rate));
    } else {
        MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", false);
    }

    if (cfg.exposure_auto) {
        MV_CC_SetEnumValue(handle_, "ExposureAuto", 2);
    } else {
        MV_CC_SetEnumValue(handle_, "ExposureAuto", 0);
        MV_CC_SetFloatValue(handle_, "ExposureTime", cfg.exposure_time);
    }

    if (cfg.gain_auto) {
        MV_CC_SetEnumValue(handle_, "GainAuto", 2);
    } else {
        MV_CC_SetEnumValue(handle_, "GainAuto", 0);
        MV_CC_SetFloatValue(handle_, "Gain", cfg.gain);
    }

    if (cfg.white_balance_mode == "manual") {
        MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", 0);
        MV_CC_SetEnumValue(handle_, "BalanceRatioSelector", 0);
        MV_CC_SetIntValue(handle_, "BalanceRatio", static_cast<int>(cfg.white_balance_red * 1000));
        MV_CC_SetEnumValue(handle_, "BalanceRatioSelector", 2);
        MV_CC_SetIntValue(handle_, "BalanceRatio", static_cast<int>(cfg.white_balance_blue * 1000));
    } else {
        MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", 1);
    }

    nRet = MV_CC_SetGrabStrategy(handle_, MV_GrabStrategy_LatestImagesOnly);
    if (nRet != MV_OK) {
        if (error) *error = "设置抓取策略失败";
        shutdown();
        return false;
    }

    nRet = MV_CC_StartGrabbing(handle_);
    if (nRet != MV_OK) {
        if (error) *error = "开始取流失败";
        shutdown();
        return false;
    }

    spdlog::info("Camera stream started.");
    return true;
}

bool HikCameraNode::grab(cv::Mat &bgr, std::string *error) {
    if (handle_ == nullptr) {
        if (error) *error = "camera not initialized";
        return false;
    }

    MV_FRAME_OUT frame = {};
    const int nRet = MV_CC_GetImageBuffer(handle_, &frame, 1000);
    if (nRet != MV_OK) {
        if (error) *error = "获取图像失败";
        return false;
    }

    const int h = static_cast<int>(frame.stFrameInfo.nHeight);
    const int w = static_cast<int>(frame.stFrameInfo.nWidth);
    cv::Mat raw(h, w, CV_8UC1, frame.pBufAddr);
    cv::cvtColor(raw, bgr, bayer_cvt_code_);
    MV_CC_FreeImageBuffer(handle_, &frame);
    return true;
}

void HikCameraNode::shutdown() {
    if (handle_ != nullptr) {
        MV_CC_StopGrabbing(handle_);
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }
}

#ifndef CAMERA_NODE_LIBRARY
int main() {
    app::logging::InitAsyncLogging();

    HikCameraNode camera;
    std::string error;
    if (!camera.init(&error)) {
        spdlog::error("Camera init failed: {}", error);
        return -1;
    }

    const std::string windowName = "Camera";
    cv::Mat img;
    int frameCount = 0;
    double fps = 0.0;
    auto fpsStart = std::chrono::steady_clock::now();

    while (true) {
        if (!camera.grab(img, &error)) {
            spdlog::error("Grab failed: {}", error);
            continue;
        }

        ++frameCount;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsStart).count();
        if (elapsedMs >= 1000) {
            fps = frameCount * 1000.0 / static_cast<double>(elapsedMs);
            frameCount = 0;
            fpsStart = now;
        }
        if (fps > 0.0) {
            cv::setWindowTitle(windowName, windowName + " | FPS: " + std::to_string(static_cast<int>(fps + 0.5)));
        }

        cv::imshow(windowName, img);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    camera.shutdown();
    cv::destroyAllWindows();
    return 0;
}
#endif