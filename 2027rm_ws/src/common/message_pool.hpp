#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "memory_layout.hpp"

template <typename T>
class SharedParamPool
{
public:
    void reserve(std::size_t count)
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->owned.reserve(count);
        state_->free.reserve(count);
    }

    void preallocate(std::size_t count)
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->owned.size() >= count)
        {
            return;
        }
        state_->owned.reserve(count);
        state_->free.reserve(count);
        while (state_->owned.size() < count)
        {
            state_->owned.emplace_back(std::make_unique<T>());
            state_->free.push_back(state_->owned.back().get());
        }
    }

    std::shared_ptr<T> acquire()
    {
        T *ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (!state_->free.empty())
            {
                ptr = state_->free.back();
                state_->free.pop_back();
            }
            else
            {
                state_->owned.emplace_back(std::make_unique<T>());
                ptr = state_->owned.back().get();
            }
        }

        std::shared_ptr<State> state = state_;
        return std::shared_ptr<T>(ptr, [state](T *released) {
            *released = T();
            std::lock_guard<std::mutex> lock(state->mutex);
            state->free.push_back(released);
        });
    }

    std::size_t pooledCount() const
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->owned.size();
    }

private:
    struct alignas(app::memory::kCacheLineSize) State
    {
        mutable std::mutex mutex;
        std::vector<std::unique_ptr<T>> owned;
        std::vector<T *> free;
    };

    std::shared_ptr<State> state_ = std::make_shared<State>();
};
