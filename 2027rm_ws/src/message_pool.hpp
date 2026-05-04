#pragma once

#include <memory>
#include <mutex>
#include <vector>

template <typename T>
class SharedParamPool
{
public:
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
                state_->owned.emplace_back(new T());
                ptr = state_->owned.back().get();
            }
        }

        *ptr = T();
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
    struct State
    {
        mutable std::mutex mutex;
        std::vector<std::unique_ptr<T>> owned;
        std::vector<T *> free;
    };

    std::shared_ptr<State> state_ = std::make_shared<State>();
};
