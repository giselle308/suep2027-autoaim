#include <chrono>
#include <cctype>
#include <cstring>
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

static const char *PixelFormatName(unsigned int fmt)
{
    switch (fmt) {
        case PixelType_Gvsp_BayerBG8: return "BayerBG8";
        case PixelType_Gvsp_BayerGB8: return "BayerGB8";
        case PixelType_Gvsp_BayerGR8: return "BayerGR8";
        case PixelType_Gvsp_BayerRG8: return "BayerRG8";
        case PixelType_Gvsp_BGR8_Packed: return "BGR8_Packed";
        case PixelType_Gvsp_RGB8_Packed: return "RGB8_Packed";
        case PixelType_Gvsp_Mono8: return "Mono8";
        case PixelType_Gvsp_HB_BayerBG8: return "HB_BayerBG8";
        case PixelType_Gvsp_HB_BayerGB8: return "HB_BayerGB8";
        case PixelType_Gvsp_HB_BayerGR8: return "HB_BayerGR8";
        case PixelType_Gvsp_HB_BayerRG8: return "HB_BayerRG8";
        case PixelType_Gvsp_HB_BGR8_Packed: return "HB_BGR8_Packed";
        default: return "Unknown";
    }
}

static std::string DescribePixelFormatSupport(void *handle)
{
    MVCC_ENUMVALUE enum_value{};
    const int ret = MV_CC_GetEnumValue(handle, "PixelFormat", &enum_value);
    if (ret != MV_OK) {
        return "unavailable";
    }

    std::string desc = "current=" + std::string(PixelFormatName(enum_value.nCurValue)) +
                       "(" + std::to_string(enum_value.nCurValue) + ")";
    if (enum_value.nSupportedNum > 0) {
        desc += " supported=";
        const unsigned int limit = std::min(enum_value.nSupportedNum, 8U);
        for (unsigned int i = 0; i < limit; ++i) {
            if (i > 0) {
                desc += ",";
            }
            const unsigned int value = enum_value.nSupportValue[i];
            desc += std::string(PixelFormatName(value)) + "(" + std::to_string(value) + ")";
        }
        if (enum_value.nSupportedNum > limit) {
            desc += ",...";
        }
    }
    return desc;
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

    nRet = MV_CC_SetIntValue(handle_, "Width", cfg.width);
    if (nRet != MV_OK) {
        if (error) *error = "设置宽度失败";
        shutdown();
        return false;
    }
    nRet = MV_CC_SetIntValue(handle_, "Height", cfg.height);
    if (nRet != MV_OK) {
        if (error) *error = "设置高度失败";
        shutdown();
        return false;
    }
    nRet = MV_CC_SetEnumValue(handle_, "PixelFormat", GetPixelFormatEnum(cfg.pixel_format));
    if (nRet != MV_OK) {
        spdlog::warn("Set PixelFormat={} failed, fallback to camera current format. {}",
                     cfg.pixel_format,
                     DescribePixelFormatSupport(handle_));
    }

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

    (void)MV_CC_SetBayerCvtQuality(handle_, 1);

    nRet = MV_CC_StartGrabbing(handle_);
    if (nRet != MV_OK) {
        if (error) *error = "开始取流失败";
        shutdown();
        return false;
    }

    spdlog::info("Camera stream started.");
    return true;
}

bool HikCameraNode::grab(cv::Mat &bgr,
                         std::chrono::steady_clock::time_point *capture_tp,
                         std::string *error) {
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
    if (capture_tp) {
        *capture_tp = std::chrono::steady_clock::now();
    }

    const int h = static_cast<int>(frame.stFrameInfo.nHeight);
    const int w = static_cast<int>(frame.stFrameInfo.nWidth);
    const auto src_pixel_type = static_cast<MvGvspPixelType>(frame.stFrameInfo.enPixelType);

    bgr.create(h, w, CV_8UC3);
    if (src_pixel_type == PixelType_Gvsp_BGR8_Packed || src_pixel_type == PixelType_Gvsp_HB_BGR8_Packed) {
        std::memcpy(bgr.data, frame.pBufAddr, static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 3U);
    } else {
        MV_CC_PIXEL_CONVERT_PARAM_EX convert_param{};
        convert_param.nWidth = static_cast<unsigned int>(w);
        convert_param.nHeight = static_cast<unsigned int>(h);
        convert_param.enSrcPixelType = src_pixel_type;
        convert_param.pSrcData = static_cast<unsigned char *>(frame.pBufAddr);
        convert_param.nSrcDataLen = frame.stFrameInfo.nFrameLen;
        convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        convert_param.pDstBuffer = bgr.data;
        convert_param.nDstBufferSize = static_cast<unsigned int>(bgr.total() * bgr.elemSize());

        const int convert_ret = MV_CC_ConvertPixelTypeEx(handle_, &convert_param);
        if (convert_ret != MV_OK) {
            MV_CC_FreeImageBuffer(handle_, &frame);
            if (error) *error = "像素格式转换失败";
            return false;
        }
    }
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
        if (!camera.grab(img, nullptr, &error)) {
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
