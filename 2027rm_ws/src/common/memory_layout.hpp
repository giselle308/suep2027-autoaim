#pragma once

#include <cstddef>
#include <cstdint>
namespace app::memory {

inline constexpr std::size_t kCacheLineSize = 64;

inline bool IsAligned(const void *ptr, std::size_t alignment = kCacheLineSize)
{
    return ptr != nullptr && (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

}  // namespace app::memory
