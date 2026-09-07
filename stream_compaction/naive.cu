#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

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

            scan_gpu(n, dev_odata, dev_idata);

            // Copy output to CPU
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free buffers
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            timer().endGpuTimer();
        }

        __host__ __device__ int divup(int dividend, int divisor) {
            return (dividend + divisor - 1) / divisor;
        }

        __host__ void pick_block_size(int n, int* numBlocks, int* blockSize) {
            *blockSize = 1024;
            *numBlocks = divup(n, *blockSize);
        }

        __global__ void kernScan(int n, int* dev_odata, int* dev_idata) {

        }

        void scan_gpu(int n, int* dev_odata, const int* dev_idata) {
            int numBlocks = 0;
            int blockSize = 0;
            pick_block_size(n, &numBlocks, &blockSize);


        }
    }
}
