#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

#include <MvCameraControl.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "camera_node.hpp"
#include "logging.hpp"
#include "profiling.hpp"

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
    if (f == "bgr8" || f == "bgr8_packed" || f == "bgr 8" || f == "brg8") return PixelType_Gvsp_BGR8_Packed;
    if (f == "rgb8" || f == "rgb8_packed") return PixelType_Gvsp_RGB8_Packed;
    if (f == "mono8") return PixelType_Gvsp_Mono8;
    if (f == "bayerbg8") return PixelType_Gvsp_BayerBG8;
    if (f == "bayergb8") return PixelType_Gvsp_BayerGB8;
    if (f == "bayergr8") return PixelType_Gvsp_BayerGR8;
    if (f == "bayerrg8") return PixelType_Gvsp_BayerRG8;
    if (f == "hb_bgr8" || f == "hb_bgr8_packed") return PixelType_Gvsp_HB_BGR8_Packed;
    if (f == "hb_bayerbg8") return PixelType_Gvsp_HB_BayerBG8;
    if (f == "hb_bayergb8") return PixelType_Gvsp_HB_BayerGB8;
    if (f == "hb_bayergr8") return PixelType_Gvsp_HB_BayerGR8;
    if (f == "hb_bayerrg8") return PixelType_Gvsp_HB_BayerRG8;
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

static bool IsPlainBayer8(MvGvspPixelType fmt)
{
    return fmt == PixelType_Gvsp_BayerBG8 ||
           fmt == PixelType_Gvsp_BayerGB8 ||
           fmt == PixelType_Gvsp_BayerGR8 ||
           fmt == PixelType_Gvsp_BayerRG8;
}

static int OpenCvBayerToBgrCode(MvGvspPixelType fmt)
{
    switch (fmt) {
        case PixelType_Gvsp_BayerBG8: return cv::COLOR_BayerBG2BGR;
        case PixelType_Gvsp_BayerGB8: return cv::COLOR_BayerGB2BGR;
        case PixelType_Gvsp_BayerGR8: return cv::COLOR_BayerGR2BGR;
        case PixelType_Gvsp_BayerRG8: return cv::COLOR_BayerRG2BGR;
        default: return -1;
    }
}

static bool IsAligned(const void *ptr, std::size_t alignment)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

static std::string HexRet(int ret)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase << ret;
    return oss.str();
}

static std::vector<std::string> PixelFormatStringCandidates(const std::string &fmt)
{
    const std::string f = ToLower(fmt);
    if (f == "bgr8" || f == "bgr8_packed" || f == "bgr 8" || f == "brg8") {
        return {"BGR8Packed", "BGR8", "BGR8_Packed", "BGR 8"};
    }
    if (f == "rgb8" || f == "rgb8_packed" || f == "rgb 8") {
        return {"RGB8Packed", "RGB8", "RGB8_Packed", "RGB 8"};
    }
    if (f == "mono8" || f == "mono 8") {
        return {"Mono8", "Mono 8"};
    }
    if (f == "bayerrg8") return {"BayerRG8"};
    if (f == "bayerbg8") return {"BayerBG8"};
    if (f == "bayergb8") return {"BayerGB8"};
    if (f == "bayergr8") return {"BayerGR8"};
    return {fmt};
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

static bool SetPixelFormatWithFallbacks(void *handle, const std::string &fmt)
{
    const unsigned int requested = GetPixelFormatEnum(fmt);
    int ret = MV_CC_SetEnumValue(handle, "PixelFormat", requested);
    if (ret == MV_OK) {
        return true;
    }

    spdlog::warn("Set PixelFormat={} by enum {} failed ret={}. {}",
                 fmt,
                 requested,
                 HexRet(ret),
                 DescribePixelFormatSupport(handle));

    for (const std::string &candidate : PixelFormatStringCandidates(fmt)) {
        ret = MV_CC_SetEnumValueByString(handle, "PixelFormat", candidate.c_str());
        if (ret == MV_OK) {
            spdlog::info("Set PixelFormat={} by string '{}' succeeded. {}",
                         fmt,
                         candidate,
                         DescribePixelFormatSupport(handle));
            return true;
        }
        spdlog::warn("Set PixelFormat={} by string '{}' failed ret={}.",
                     fmt,
                     candidate,
                     HexRet(ret));
    }
    return false;
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
    alignment_logged_ = false;
    spdlog::info("Pixel conversion backend=opencv");

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

    spdlog::info("Initial PixelFormat support: {}", DescribePixelFormatSupport(handle_));
    if (!SetPixelFormatWithFallbacks(handle_, cfg.pixel_format)) {
        spdlog::warn("Set PixelFormat={} failed with all methods, fallback to camera current format. {}",
                     cfg.pixel_format,
                     DescribePixelFormatSupport(handle_));
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

bool HikCameraNode::grab(cv::Mat &bgr,
                         std::chrono::steady_clock::time_point *capture_tp,
                         double *grab_ms,
                         double *pixel_convert_ms,
                         std::string *error) {
    app::profiling::ScopedTimer grab_timer(app::profiling::Stage::CameraGrab);
    const auto grab_start = std::chrono::steady_clock::now();
    if (grab_ms) {
        *grab_ms = 0.0;
    }
    if (pixel_convert_ms) {
        *pixel_convert_ms = 0.0;
    }
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
    if (grab_ms) {
        const auto grab_end = capture_tp ? *capture_tp : std::chrono::steady_clock::now();
        *grab_ms = std::chrono::duration<double, std::milli>(grab_end - grab_start).count();
    }

    const int h = static_cast<int>(frame.stFrameInfo.nHeight);
    const int w = static_cast<int>(frame.stFrameInfo.nWidth);
    const auto src_pixel_type = static_cast<MvGvspPixelType>(frame.stFrameInfo.enPixelType);

    bgr.create(h, w, CV_8UC3);
    const auto convert_start = std::chrono::steady_clock::now();
    app::profiling::ScopedTimer convert_timer(app::profiling::Stage::PixelConvert);
    if (src_pixel_type == PixelType_Gvsp_BGR8_Packed || src_pixel_type == PixelType_Gvsp_HB_BGR8_Packed) {
        std::memcpy(bgr.data, frame.pBufAddr, static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 3U);
    } else if (src_pixel_type == PixelType_Gvsp_RGB8_Packed) {
        const cv::Mat rgb(h, w, CV_8UC3, frame.pBufAddr);
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    } else if (src_pixel_type == PixelType_Gvsp_Mono8) {
        const cv::Mat mono(h, w, CV_8UC1, frame.pBufAddr);
        cv::cvtColor(mono, bgr, cv::COLOR_GRAY2BGR);
    } else if (IsPlainBayer8(src_pixel_type)) {
        const cv::Mat raw(h, w, CV_8UC1, frame.pBufAddr);
        cv::cvtColor(raw, bgr, OpenCvBayerToBgrCode(src_pixel_type));
    } else {
        MV_CC_FreeImageBuffer(handle_, &frame);
        if (error) {
            *error = std::string("unsupported pixel format for OpenCV conversion: ") +
                     PixelFormatName(static_cast<unsigned int>(src_pixel_type));
        }
        return false;
    }
    if (!alignment_logged_) {
        spdlog::info("Frame buffer alignment: raw_64B={} bgr_64B={} bgr_step={} continuous={}",
                     IsAligned(frame.pBufAddr, 64),
                     IsAligned(bgr.data, 64),
                     bgr.step,
                     bgr.isContinuous());
        alignment_logged_ = true;
    }
    if (pixel_convert_ms) {
        *pixel_convert_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - convert_start).count();
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
        if (!camera.grab(img, nullptr, nullptr, nullptr, &error)) {
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
