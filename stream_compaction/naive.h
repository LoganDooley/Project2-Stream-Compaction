#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanGpu(int n, int* dev_odata, int* dev_idata);

        __global__ void kernScanBlock(int n, int* dev_data, int* dev_blockSums);

        __host__ int getSharedMemorySize(int blockSize);
    }
}
