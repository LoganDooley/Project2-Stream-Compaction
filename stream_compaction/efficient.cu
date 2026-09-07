#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int pow_of_two = ilog2ceil(n);
            int n_new = ipow2(pow_of_two);
            // Allocate buffer
            int* dev_data;
            cudaMalloc((void**)&dev_data, n_new * sizeof(int));

            // Initialize with 0s
            cudaMemset(dev_data, 0, n_new * sizeof(int));

            // Copy input to CPU
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            scan_gpu(n_new, dev_data);

            // Copy output to CPU
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free buffers
            cudaFree(dev_data);
            timer().endGpuTimer();
        }

        void scan_gpu(int n, int* dev_data)
        {
            scan_gpu_upsweep(n, dev_data);
            scan_gpu_downsweep(n, dev_data);
        }

        __host__ __device__ int divup(int dividend, int divisor) {
            return (dividend + divisor - 1) / divisor;
        }

        __host__ void pick_block_size(int n, int* numBlocks, int* blockSize) {
            *blockSize = 1024;
            *numBlocks = divup(n, *blockSize);
        }

        __global__ void kernUpsweep(int n, int d, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int half_stride = 1 << d;
            int stride = 1 << (d + 1);

            int k = index * stride;

            dev_data[k + stride - 1] = dev_data[k + half_stride - 1] + dev_data[k + stride - 1];
        }

        void scan_gpu_upsweep(int n, int* dev_data) {
            int d_max = ilog2(n) - 1;
            int num_threads = n / 2;
            for (int d = 0; d <= d_max; d++) {
                int numBlocks = 0;
                int blockSize = 0;
                pick_block_size(num_threads, &numBlocks, &blockSize);

                kernUpsweep << <numBlocks, blockSize >> > (num_threads, d, dev_data);

                num_threads /= 2;
            }
        }

        __global__ void kernDownsweep(int n, int d, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int half_stride = 1 << d;
            int stride = 1 << (d + 1);

            int k = index * stride;

            int t = dev_data[k + half_stride - 1];
            dev_data[k + half_stride - 1] = dev_data[k + stride - 1];
            dev_data[k + stride - 1] = t + dev_data[k + stride - 1];
        }

        void scan_gpu_downsweep(int n, int* dev_data) {
            cudaMemset(dev_data + (n - 1), 0, sizeof(int));

            int d_max = ilog2(n) - 1;
            int num_threads = 1;
            for (int d = d_max; d >= 0; d--) {
                int numBlocks = 0;
                int blockSize = 0;
                pick_block_size(num_threads, &numBlocks, &blockSize);

                kernDownsweep << <numBlocks, blockSize >> > (num_threads, d, dev_data);

                num_threads *= 2;
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
