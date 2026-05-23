#pragma once

#include <cstdlib>

#include <opencv2/core.hpp>

#include "memory_layout.hpp"

namespace app::memory {

class AlignedMatAllocator final : public cv::MatAllocator
{
public:
    cv::UMatData *allocate(int dims,
                           const int *sizes,
                           int type,
                           void *data,
                           std::size_t *step,
                           cv::AccessFlag,
                           cv::UMatUsageFlags) const override
    {
        const std::size_t elem_size = CV_ELEM_SIZE(type);
        std::size_t total = elem_size;
        if (step)
        {
            for (int i = dims - 1; i >= 0; --i)
            {
                step[i] = total;
                total *= static_cast<std::size_t>(sizes[i]);
            }
        }
        else
        {
            for (int i = dims - 1; i >= 0; --i)
            {
                total *= static_cast<std::size_t>(sizes[i]);
            }
        }

        auto *u = new cv::UMatData(this);
        if (data)
        {
            u->data = u->origdata = static_cast<uchar *>(data);
            u->size = total;
            u->flags |= cv::UMatData::USER_ALLOCATED;
            return u;
        }

        void *ptr = nullptr;
        if (posix_memalign(&ptr, kCacheLineSize, total) != 0)
        {
            delete u;
            CV_Error(cv::Error::StsNoMem, "posix_memalign failed for aligned cv::Mat allocation");
        }

        u->data = u->origdata = static_cast<uchar *>(ptr);
        u->size = total;
        return u;
    }

    bool allocate(cv::UMatData *, cv::AccessFlag, cv::UMatUsageFlags) const override
    {
        return false;
    }

    void deallocate(cv::UMatData *u) const override
    {
        if (!u)
        {
            return;
        }
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            std::free(u->origdata);
        }
        delete u;
    }
};

inline AlignedMatAllocator &GetAlignedMatAllocator()
{
    static AlignedMatAllocator allocator;
    return allocator;
}

inline void CreateAlignedMat(cv::Mat &mat, int rows, int cols, int type)
{
    if (!mat.empty() &&
        mat.rows == rows &&
        mat.cols == cols &&
        mat.type() == type &&
        IsAligned(mat.data))
    {
        return;
    }

    mat.release();
    mat.allocator = &GetAlignedMatAllocator();
    mat.create(rows, cols, type);
}

}  // namespace app::memory
