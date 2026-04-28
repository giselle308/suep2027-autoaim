#include "profiling.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>

#include <spdlog/spdlog.h>

namespace app::profiling {
namespace {

struct StageStat
{
    uint64_t count = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
};

constexpr std::array<const char *, static_cast<std::size_t>(Stage::Count)> kStageNames = {
    "camera_grab",
    "pixel_convert",
    "preprocess",
    "infer_async",
    "postprocess",
    "pnp_solve",
    "pnp_total"};

std::mutex g_mu;
std::array<StageStat, static_cast<std::size_t>(Stage::Count)> g_stats{};
std::atomic<bool> g_enabled(false);
int g_interval_ms = 1000;
std::chrono::steady_clock::time_point g_last_log = std::chrono::steady_clock::now();

std::size_t ToIndex(Stage stage)
{
    return static_cast<std::size_t>(stage);
}

}  // namespace

void Configure(bool enable, int interval_ms)
{
    std::lock_guard<std::mutex> lk(g_mu);
    g_enabled.store(enable, std::memory_order_release);
    g_interval_ms = std::max(100, interval_ms);
    g_last_log = std::chrono::steady_clock::now();
    for (auto &stat : g_stats)
    {
        stat = StageStat{};
    }
}

bool Enabled()
{
    return g_enabled.load(std::memory_order_acquire);
}

void Record(Stage stage, double ms)
{
    if (!g_enabled.load(std::memory_order_acquire))
    {
        return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    const std::size_t idx = ToIndex(stage);
    if (idx >= g_stats.size())
    {
        return;
    }
    StageStat &stat = g_stats[idx];
    ++stat.count;
    stat.total_ms += ms;
    stat.max_ms = std::max(stat.max_ms, ms);
}

void LogIfDue()
{
    std::array<StageStat, static_cast<std::size_t>(Stage::Count)> snapshot{};
    int interval_ms = 1000;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_enabled.load(std::memory_order_acquire))
        {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_last_log).count();
        if (elapsed_ms < g_interval_ms)
        {
            return;
        }
        snapshot = g_stats;
        for (auto &stat : g_stats)
        {
            stat = StageStat{};
        }
        interval_ms = g_interval_ms;
        g_last_log = now;
    }

    std::string msg = "[PROF]";
    bool has_data = false;
    for (std::size_t i = 0; i < snapshot.size(); ++i)
    {
        const StageStat &stat = snapshot[i];
        if (stat.count == 0)
        {
            continue;
        }
        has_data = true;
        const double avg_ms = stat.total_ms / static_cast<double>(stat.count);
        msg += " ";
        msg += kStageNames[i];
        msg += "_avg=" + std::to_string(avg_ms);
        msg += " ";
        msg += kStageNames[i];
        msg += "_max=" + std::to_string(stat.max_ms);
        msg += " ";
        msg += kStageNames[i];
        msg += "_n=" + std::to_string(stat.count);
    }
    if (has_data)
    {
        spdlog::info("{} interval_ms={}", msg, interval_ms);
    }
}

ScopedTimer::ScopedTimer(Stage stage)
    : stage_(stage),
      enabled_(Enabled()),
      start_(std::chrono::steady_clock::now())
{
}

ScopedTimer::~ScopedTimer()
{
    if (!enabled_)
    {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    Record(stage_, ms);
}

}  // namespace app::profiling
