#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanGpu(int n, int* dev_data);

        void scanGpuUpsweep(int n, int* dev_data);

        void scanGpuDownsweep(int n, int* dev_data);

        int compact(int n, int *odata, const int *idata);

        void mapToBooleanGpu(int n, int* dev_bools, const int* dev_idata);

        void scatterGpu(int n, int* dev_odata,
            const int* dev_idata, const int* dev_bools, const int* dev_indices);

        __global__ void kernScanBlock(int n, int* dev_data, int* dev_blockSums);
    }
}
