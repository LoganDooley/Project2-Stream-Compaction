#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <cmath>
#include <algorithm>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // Allocate buffers
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // Copy input to CPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_result = scan_gpu(n, dev_odata, dev_idata);

            // Copy output to CPU
            cudaMemcpy(odata, dev_result, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free buffers
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            timer().endGpuTimer();
        }

        __host__ __device__ int divup(int dividend, int divisor) {
            return (dividend + divisor - 1) / divisor;
        }

        __host__ void pick_block_size(int n, int* numBlocks, int* blockSize) {
            *blockSize = 1024;
            *numBlocks = divup(n, *blockSize);
        }

        __global__ void kernRightShift(int n, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (index >= n) {
                return;
            }

            dev_data[index] = index == 0 ? 0 : dev_data[index - 1];
        }

        __global__ void kernScan(int n, int offset, int* dev_odata, const int* dev_idata) {
            int k = blockDim.x * blockIdx.x + threadIdx.x;

            if (k >= n) {
                return;
            }

            if (k >= offset) {
                dev_odata[k] = dev_idata[k - offset] + dev_idata[k];
            }
            else {
                dev_odata[k] = dev_idata[k];
            }
        }

        int* scan_gpu(int n, int* dev_odata, int* dev_idata) {
            int numBlocks = 0;
            int blockSize = 0;
            pick_block_size(n, &numBlocks, &blockSize);

            // Right shift the input to do an exclusive rather than inclusive scan
            kernRightShift << <numBlocks, blockSize >> > (n, dev_idata);

            // Perform double buffered scan algorithm
            int d_max = ilog2ceil(n);

            int offset = 1;
            for (int d = 1; d <= d_max; d++) {
                kernScan << <numBlocks, blockSize >> > (n, offset, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
                offset *= 2;
            }

            return dev_idata;
        }
    }
}
