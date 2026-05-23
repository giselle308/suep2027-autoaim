#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace app {
namespace logging {

struct LogConfig {
    std::string level = "info";
    std::string flush_level = "warn";
    std::size_t queue_size = 8192;
    std::size_t thread_count = 1;
    std::string overflow_policy = "overrun_oldest";
    std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [tid %t] %v";
};

inline spdlog::level::level_enum ParseLogLevel(const std::string &name, spdlog::level::level_enum fallback) {
    if (name == "trace") return spdlog::level::trace;
    if (name == "debug") return spdlog::level::debug;
    if (name == "info") return spdlog::level::info;
    if (name == "warn" || name == "warning") return spdlog::level::warn;
    if (name == "error") return spdlog::level::err;
    if (name == "critical") return spdlog::level::critical;
    if (name == "off") return spdlog::level::off;
    return fallback;
}

inline spdlog::async_overflow_policy ParseOverflowPolicy(const std::string &name) {
    if (name == "block") {
        return spdlog::async_overflow_policy::block;
    }
    return spdlog::async_overflow_policy::overrun_oldest;
}

inline void InitAsyncLogging(const LogConfig &cfg = LogConfig()) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [&cfg]() {
        const std::size_t queue_size = (cfg.queue_size < 1024) ? 1024 : cfg.queue_size;
        const std::size_t thread_count = (cfg.thread_count < 1) ? 1 : cfg.thread_count;
        spdlog::init_thread_pool(queue_size, thread_count);

        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto logger = std::make_shared<spdlog::async_logger>(
            "rm_async",
            sink,
            spdlog::thread_pool(),
            ParseOverflowPolicy(cfg.overflow_policy));

        logger->set_level(ParseLogLevel(cfg.level, spdlog::level::info));
        logger->flush_on(ParseLogLevel(cfg.flush_level, spdlog::level::warn));
        spdlog::set_default_logger(logger);
        spdlog::set_pattern(cfg.pattern.empty() ? "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [tid %t] %v" : cfg.pattern);
    });
}

inline void ApplyRuntimeLoggingConfig(const LogConfig &cfg) {
    auto logger = spdlog::default_logger();
    if (!logger) {
        return;
    }

    logger->set_level(ParseLogLevel(cfg.level, logger->level()));
    logger->flush_on(ParseLogLevel(cfg.flush_level, spdlog::level::warn));
    if (!cfg.pattern.empty()) {
        spdlog::set_pattern(cfg.pattern);
    }
}

}  // namespace logging
}  // namespace app
