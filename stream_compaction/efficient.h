#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scan_gpu(int n, int* dev_data);

        void scan_gpu_upsweep(int n, int* dev_data);

        void scan_gpu_downsweep(int n, int* dev_data);

        int compact(int n, int *odata, const int *idata);

        void map_to_boolean_gpu(int n, int* dev_bools, const int* dev_idata);

        void scatter_gpu(int n, int* dev_odata,
            const int* dev_idata, const int* dev_bools, const int* dev_indices);
    }
}
