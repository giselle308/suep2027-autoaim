#pragma once

#include <string>

namespace app::runtime {

void ApplyThreadAffinity(const char *node_name, int cpu);
std::string AffinityDescription(int cpu);

}  // namespace app::runtime
