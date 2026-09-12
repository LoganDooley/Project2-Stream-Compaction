#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

inline int ipow2(unsigned int x) {
    return 1 << x;
}

inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

namespace StreamCompaction {
    namespace Common {
        __global__ void kernIncrementByBlockSums(int chunkSize, int n, int* dev_odata, const int* dev_blockSums);

        template <typename KernelFunction>
        void pickBlockSize(KernelFunction kernel, int n, int* numBlocks, int* blockSize) {
            int minGridSize = 0;
            int bestBlockSize = 0;

            // Find what CUDA would recommend
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, kernel, 0, 0);
			checkCUDAError("cudaOccupancyMaxPotentialBlockSize failed");

            // If our n is smaller than what cuda determines is the recommended size,
            // find a multiple of 32 that fits
            if (n < bestBlockSize) {
                bestBlockSize = std::max(32, (n / 32) * 32);
            }

            *blockSize = bestBlockSize;
            *numBlocks = (n + *blockSize - 1) / *blockSize;
        }

        template <typename KernelFunction>
        void pickBlockSizePowOfTwo(KernelFunction kernel, int n, int* numBlocks, int* blockSize) {
            int minGridSize = 0;
            int bestBlockSize = 0;

            // Find what CUDA would recommend
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, kernel, 0, 0);
			checkCUDAError("cudaOccupancyMaxPotentialBlockSize failed");

            // If the best block size isn't a power of 2, round down to the nearest power of 2
            if ((bestBlockSize & (bestBlockSize - 1)) != 0) {
                int nextSmallestPowerOfTwo = 1;
                while (nextSmallestPowerOfTwo * 2 <= bestBlockSize) {
                    nextSmallestPowerOfTwo *= 2;
                }
                bestBlockSize = nextSmallestPowerOfTwo;
            }

            // If our n is smaller than the recommended block size, 
            // find a power of 2 that fits
            if (n < bestBlockSize) {
                int fallback = 32; // Start from minimum warp size
                while (fallback < n && fallback < bestBlockSize) {
                    fallback *= 2;
                }
                bestBlockSize = fallback;
            }

            *blockSize = bestBlockSize;
            *numBlocks = (n + *blockSize - 1) / *blockSize;
        }

        template <typename KernelFunc, typename ChunkSizeFunc, typename SharedMemorySizeFunc>
        void scanRecursive(KernelFunc kernScanBlock, ChunkSizeFunc chunkSizeFunc, SharedMemorySizeFunc sharedMemorySizeFunc, int elementsPerThread, int n, int* dev_data) {
            if (n <= 0) {
                return;
            }

            int numChunks = 0;
            int chunkSize = 0;
            chunkSizeFunc(kernScanBlock, n, &numChunks, &chunkSize);

			int threadsPerChunk = divup(chunkSize, elementsPerThread);

            size_t sharedMemorySize = sharedMemorySizeFunc(chunkSize);

            if (numChunks <= 1) {
                // Base case
                kernScanBlock << <numChunks, threadsPerChunk, sharedMemorySize >> > (chunkSize, n, dev_data, nullptr);
				checkCUDAError("kernScanBlock failed");
                return;
            }

            // Allocate buffer for chunk sums
            int* dev_chunkSums;
            cudaMalloc((void**)&dev_chunkSums, numChunks * sizeof(int));
			checkCUDAError("cudaMalloc dev_chunkSums failed");

            // Scan each chunk separately
            kernScanBlock << <numChunks, threadsPerChunk, sharedMemorySize >> > (chunkSize, n, dev_data, dev_chunkSums);
			checkCUDAError("kernScanBlock failed");

            // Scan the chunk sums
            scanRecursive(kernScanBlock, chunkSizeFunc, sharedMemorySizeFunc, elementsPerThread, numChunks, dev_chunkSums);

            // Increment sums by the chunk sums
            kernIncrementByBlockSums << <numChunks, chunkSize >> > (chunkSize, n, dev_data, dev_chunkSums);
			checkCUDAError("kernIncrementByBlockSums failed");

            cudaFree(dev_chunkSums);
			checkCUDAError("cudaFree dev_chunkSums failed");
        }

        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

        /**
        * This class is used for timing the performance
        * Uncopyable and unmovable
        *
        * Adapted from WindyDarian(https://github.com/WindyDarian)
        */
        class PerformanceTimer
        {
        public:
            PerformanceTimer()
            {
                cudaEventCreate(&event_start);
                cudaEventCreate(&event_end);
            }

            ~PerformanceTimer()
            {
                cudaEventDestroy(event_start);
                cudaEventDestroy(event_end);
            }

            void startCpuTimer()
            {
                if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
                cpu_timer_started = true;

                time_start_cpu = std::chrono::high_resolution_clock::now();
            }

            void endCpuTimer()
            {
                time_end_cpu = std::chrono::high_resolution_clock::now();

                if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

                std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
                prev_elapsed_time_cpu_milliseconds =
                    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

                cpu_timer_started = false;
            }

            void startGpuTimer()
            {
                if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
                gpu_timer_started = true;

                cudaEventRecord(event_start);
            }

            void endGpuTimer()
            {
                cudaEventRecord(event_end);
                cudaEventSynchronize(event_end);

                if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

                cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
                gpu_timer_started = false;
            }

            float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
            {
                return prev_elapsed_time_cpu_milliseconds;
            }

            float getGpuElapsedTimeForPreviousOperation() //noexcept
            {
                return prev_elapsed_time_gpu_milliseconds;
            }

            // remove copy and move functions
            PerformanceTimer(const PerformanceTimer&) = delete;
            PerformanceTimer(PerformanceTimer&&) = delete;
            PerformanceTimer& operator=(const PerformanceTimer&) = delete;
            PerformanceTimer& operator=(PerformanceTimer&&) = delete;

        private:
            cudaEvent_t event_start = nullptr;
            cudaEvent_t event_end = nullptr;

            using time_point_t = std::chrono::high_resolution_clock::time_point;
            time_point_t time_start_cpu;
            time_point_t time_end_cpu;

            bool cpu_timer_started = false;
            bool gpu_timer_started = false;

            float prev_elapsed_time_cpu_milliseconds = 0.f;
            float prev_elapsed_time_gpu_milliseconds = 0.f;
        };
    }
}
