#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "common.h"
#include "efficient.h"

#define EFFICIENT_USE_SHARED_MEMORY 1
#define EFFICIENT_USE_COMPACTED_INDICES 1

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
            int powOfTwo = ilog2ceil(n);
            int nNew = ipow2(powOfTwo);
            // Allocate buffer
            int* dev_data;
            cudaMalloc((void**)&dev_data, nNew * sizeof(int));

            // Initialize with 0s
            cudaMemset(dev_data, 0, nNew * sizeof(int));

            // Copy input to CPU
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
#if EFFICIENT_USE_SHARED_MEMORY
            Common::scanRecursive(kernScanBlock,
                Common::pickBlockSizePowOfTwo<decltype(kernScanBlock)>,
                getSharedMemorySize,
                nNew,
                dev_data);
#else
            scanGpu(n_new, dev_data);
#endif
            timer().endGpuTimer();

            // Copy output to CPU
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free buffers
            cudaFree(dev_data);
        }

        void scanGpu(int n, int* dev_data)
        {
            scanGpuUpsweep(n, dev_data);
            scanGpuDownsweep(n, dev_data);
        }

        __host__ __device__ int divup(int dividend, int divisor) {
            return (dividend + divisor - 1) / divisor;
        }

        __global__ void kernUpsweep(int dataLength, int activeThreadCount, int d, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= activeThreadCount) {
                return;
            }

            int halfStride = 1 << d;
            int stride = 1 << (d + 1);

            int k = index * stride;

            if (k + stride - 1 >= dataLength) {
                return;
            }

            dev_data[k + stride - 1] = dev_data[k + halfStride - 1] + dev_data[k + stride - 1];
        }

        void scanGpuUpsweep(int n, int* dev_data) {
            int dMax = ilog2(n) - 1;
            int numThreads = n / 2;
            for (int d = 0; d <= dMax; d++) {
                int numBlocks = 0;
                int blockSize = 0;
                Common::pickBlockSize(kernUpsweep, numThreads, &numBlocks, &blockSize);

                kernUpsweep << <numBlocks, blockSize >> > (n, numThreads, d, dev_data);

                numThreads /= 2;
            }
        }

        __global__ void kernDownsweep(int dataLength, int activeThreadCount, int d, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= activeThreadCount) {
                return;
            }

            int halfStride = 1 << d;
            int stride = 1 << (d + 1);

            int k = index * stride;

            if (k + stride - 1 >= dataLength) {
                return;
            }

            int t = dev_data[k + halfStride - 1];
            dev_data[k + halfStride - 1] = dev_data[k + stride - 1];
            dev_data[k + stride - 1] = t + dev_data[k + stride - 1];
        }

        void scanGpuDownsweep(int n, int* dev_data) {
            cudaMemset(dev_data + (n - 1), 0, sizeof(int));

            int d_max = ilog2ceil(n) - 1;
            int numThreads = 1;
            for (int d = d_max; d >= 0; d--) {
                int numBlocks = 0;
                int blockSize = 0;
                Common::pickBlockSize(kernDownsweep, numThreads, &numBlocks, &blockSize);

                kernDownsweep << <numBlocks, blockSize >> > (n, numThreads, d, dev_data);

                numThreads *= 2;
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
            if (n == 0) {
                return 0;
            }

            int powOfTwo = ilog2ceil(n);
            int nNew = ipow2(powOfTwo);

            // Allocate buffers
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, nNew * sizeof(int));
            cudaMalloc((void**)&dev_indices, nNew * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // Copy input to CPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // Convert to bools (only first n elements is necessary because of the Memset)
            mapToBooleanGpu(n, dev_bools, dev_idata);

            // Initialize indices with 0s so unset inputs are not counted in the scan
            cudaMemset(dev_indices, 0, nNew * sizeof(int));
            // Copy bools to indices since scan operates in place
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            // Scan to get indices
            scanGpu(nNew, dev_indices);

            // Scatter first n elements
            scatterGpu(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

            // Copy output data to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Copy the last index in the index buffer to host
            int lastIndex = 0;
            cudaMemcpy(&lastIndex, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            int lastValue = idata[n - 1];

            int countRemaining = lastIndex + (lastValue != 0);

            // Free buffers
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);
            cudaFree(dev_idata);
            return countRemaining;
        }

        void mapToBooleanGpu(int n, int* dev_bools, const int* dev_idata) {
            int numBlocks = 0;
            int blockSize = 0;
            Common::pickBlockSize(Common::kernMapToBoolean, n, &numBlocks, &blockSize);

            Common::kernMapToBoolean << <numBlocks, blockSize >> > (n, dev_bools, dev_idata);
        }

        void scatterGpu(int n, int* dev_odata,
            const int* dev_idata, const int* dev_bools, const int* dev_indices) {
            int numBlocks = 0;
            int blockSize = 0;
            Common::pickBlockSize(Common::kernScatter, n, &numBlocks, &blockSize);

            Common::kernScatter << <numBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
        }

        __global__ void kernScanBlock(int n, int* dev_data, int* dev_blockSums) {
            // TODO: Consider optimizing this by launching a kernel with half the block size
            // that way the first iteration of the upsweep isn't wasting half the threads
            extern __shared__ int temp[];

            __shared__ cuda::barrier<cuda::thread_scope_block> bar;
            auto block = cooperative_groups::this_thread_block();

            if (block.thread_rank() == 0)
            {
                init(&bar, block.size());
            }
            block.sync();

            int localIndex = threadIdx.x;
            int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

            // Load into shared memory
            temp[localIndex] = (globalIndex < n) ? dev_data[globalIndex] : 0;
            block.sync();

            // Upsweep
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
#if EFFICIENT_USE_COMPACTED_INDICES
                int active_threads = blockDim.x / (stride * 2);
                if(localIndex < active_threads) {
                    // Get right-most index of this given stride
                    int rightIndex = (localIndex + 1) * stride * 2 - 1;
					int leftIndex = rightIndex - stride;
                    temp[rightIndex] += temp[leftIndex];
				}
#else
                // Get right-most index of this given stride
                int index = (localIndex + 1) * stride * 2 - 1;
                if (index < blockDim.x) {
                    temp[index] += temp[index - stride];
                }
#endif
                block.sync();
            }

            if (localIndex == 0) {
                // Save block sum if needed and replace with 0 before downsweep
                if (dev_blockSums != nullptr) {
                    dev_blockSums[blockIdx.x] = temp[blockDim.x - 1];
                }
                temp[blockDim.x - 1] = 0;
            }
            block.sync();

            // Downsweep
            for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
#if EFFICIENT_USE_COMPACTED_INDICES
				int activeThreads = blockDim.x / (stride * 2);
                if(localIndex < activeThreads) {
                    // Get right-most index of this given stride
                    int rightIndex = (localIndex + 1) * stride * 2 - 1;
                    int leftIndex = rightIndex - stride;

                    int leftChild = temp[leftIndex];

                    // Swap and add
                    temp[leftIndex] = temp[rightIndex];
                    temp[rightIndex] += leftChild;
				}
#else
                // Get right most index of this given stride
                int index = (localIndex + 1) * stride * 2 - 1;
                if (index < blockDim.x) {
                    int leftChild = temp[index - stride];

                    // Copy right into left
                    temp[index - stride] = temp[index];
                    // Add left to the right
                    temp[index] += leftChild;
                }
#endif
                block.sync();
            }

            // Write back to global memory
            if (globalIndex < n) {
                dev_data[globalIndex] = temp[localIndex];
            }
        }

        __host__ int getSharedMemorySize(int blockSize)
        {
            return blockSize * sizeof(int);
        }
    }
}
