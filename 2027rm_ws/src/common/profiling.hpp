#pragma once

#include <array>
#include <chrono>
#include <cstdint>

namespace app::profiling {

enum class Stage : std::size_t
{
    CameraGrab = 0,
    PixelConvert,
    Preprocess,
    InferAsync,
    Postprocess,
    PnpSolve,
    PnpTotal,
    Count
};

void Configure(bool enable, int interval_ms);
bool Enabled();
void Record(Stage stage, double ms);
void LogIfDue();

class ScopedTimer
{
public:
    explicit ScopedTimer(Stage stage);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
    Stage stage_;
    bool enabled_ = false;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace app::profiling
