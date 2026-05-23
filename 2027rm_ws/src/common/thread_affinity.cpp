#include "thread_affinity.hpp"

#ifdef __linux__
#include <cerrno>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#endif

#include <spdlog/spdlog.h>

namespace app::runtime {

std::string AffinityDescription(int cpu)
{
    return cpu >= 0 ? std::to_string(cpu) : std::string("disabled");
}

void ApplyThreadAffinity(const char *node_name, int cpu)
{
    if (cpu < 0)
    {
        return;
    }

#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0)
    {
        spdlog::warn("[affinity] bind {} to CPU {} failed: {}", node_name, cpu, std::strerror(rc));
        return;
    }
    spdlog::info("[affinity] bound {} to CPU {}", node_name, cpu);
#else
    spdlog::warn("[affinity] bind {} to CPU {} ignored: unsupported platform", node_name, cpu);
#endif
}

}  // namespace app::runtime
