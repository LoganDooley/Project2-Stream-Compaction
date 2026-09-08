#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scan_gpu(int n, int* dev_odata, int* dev_idata);
    
        void recursive_scan_gpu(int n, int* dev_odata, const int* dev_idata);
    }
}
