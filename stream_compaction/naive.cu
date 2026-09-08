#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <cmath>
#include <algorithm>

#define NAIVE_USE_SHARED_MEMORY 1

namespace StreamCompaction {
    namespace Naive {
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
            // Allocate buffers
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // Copy input to CPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

#if NAIVE_USE_SHARED_MEMORY
            recursive_scan_gpu(n, dev_odata, dev_idata);
#else
            scan_gpu(n, dev_odata, dev_idata);
#endif

            // Copy output to CPU
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free buffers
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            timer().endGpuTimer();
        }

        __host__ __device__ int divup(int dividend, int divisor) {
            return (dividend + divisor - 1) / divisor;
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

        __global__ void kernScanRightShifted(int n, int offset, int* dev_odata, const int* dev_idata) {
            int k = blockDim.x * blockIdx.x + threadIdx.x;

            if (k >= n) {
                return;
            }

            if (k >= offset) {
                int a = k - offset - 1 >= 0 ? dev_idata[k - offset - 1] : 0;
                int b = k - 1 >= 0 ? dev_idata[k - 1] : 0;
                dev_odata[k] = a + b;
            }
            else {
                dev_odata[k] = k - 1 >= 1 ? dev_idata[k - 1] : 0;
            }
        }

        void scan_gpu(int n, int* dev_odata, int* dev_idata) {
            int numBlocks = 0;
            int blockSize = 0;
            pick_block_size(kernScan, n, &numBlocks, &blockSize);

            // Perform double buffered scan algorithm
            int d_max = ilog2ceil(n);

            int offset = 1;
            for (int d = 1; d <= d_max; d++) {
                if (d == 1) {
                    kernScanRightShifted << <numBlocks, blockSize >> > (n, offset, dev_odata, dev_idata);
                }
                else {
                    kernScan << <numBlocks, blockSize >> > (n, offset, dev_odata, dev_idata);
                }
                std::swap(dev_odata, dev_idata);
                offset *= 2;
            }

            std::swap(dev_odata, dev_idata);
        }

        __global__ void kernScanBlock(int n, int* dev_odata, const int* dev_idata, int* dev_blockSums) {
            extern __shared__ int temp[];

            int localIndex = threadIdx.x;
            int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

            // Indices to fake double buffering
            int pout = 0;
            int pin = 1;

            // Load data from global memory
            temp[pout * blockDim.x + localIndex] = (globalIndex < n) ? dev_idata[globalIndex] : 0;
            __syncthreads();

            // Do sum in shared memory by "ping-pong"ing the two halves of the shared memory segment
            for (int offset = 1; offset < blockDim.x; offset *= 2) {
                // Ping pong by flipping the out and in flags
                pout = 1 - pout;
                pin = 1 - pin;

                if (localIndex >= offset) {
                    temp[pout * blockDim.x + localIndex] = temp[pin * blockDim.x + localIndex - offset] + temp[pin * blockDim.x + localIndex];
                }
                else {
                    temp[pout * blockDim.x + localIndex] = temp[pin * blockDim.x + localIndex];
                }
                __syncthreads();
            }

            // Write back block sum 
            if (dev_blockSums != nullptr && localIndex == blockDim.x - 1) {
                dev_blockSums[blockIdx.x] = temp[pout * blockDim.x + localIndex];
            }
            __syncthreads();

            // Convert to exclusive scan and write back to global memory
            if (globalIndex < n) {
                dev_odata[globalIndex] = (localIndex == 0) ? 0 : temp[pout * blockDim.x + localIndex - 1];
            }
        }

        __global__ void kernIncrementByBlockSums(int n, int* dev_odata, const int* dev_blockSums) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            // Only blocks 1+ should increment
            if (blockIdx.x > 0) {
                dev_odata[index] += dev_blockSums[blockIdx.x];
            }
        }

        void recursive_scan_gpu(int n, int* dev_odata, const int* dev_idata) {
            if (n <= 0) {
                return;
            }
            
            int numBlocks = 0;
            int blockSize = 0;
            pick_block_size(kernScanBlock, n, &numBlocks, &blockSize);

            // 2 values per thread since it is double buffering via shared memory
            size_t sharedMemorySize = 2 * blockSize * sizeof(int);

            if (numBlocks <= 1) {
                // Base case
                kernScanBlock << <numBlocks, blockSize, sharedMemorySize >> > (n, dev_odata, dev_idata, nullptr);
                return;
            }

            // Allocate buffer for block sums
            int* dev_blockSums;
            cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));

            // Scan each block separately
            kernScanBlock << <numBlocks, blockSize, sharedMemorySize >> > (n, dev_odata, dev_idata, dev_blockSums);

            // Scan the block sums
            recursive_scan_gpu(numBlocks, dev_blockSums, dev_blockSums);

            // Increment sums by the block sums
            kernIncrementByBlockSums << <numBlocks, blockSize >> > (n, dev_odata, dev_blockSums);

            cudaFree(dev_blockSums);
        }
    }
}
