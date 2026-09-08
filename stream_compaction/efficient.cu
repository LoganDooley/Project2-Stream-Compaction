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

        __global__ void kernUpsweep(int data_length, int active_thread_count, int d, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= active_thread_count) {
                return;
            }

            int half_stride = 1 << d;
            int stride = 1 << (d + 1);

            int k = index * stride;

            if (k + stride - 1 >= data_length) {
                return;
            }

            dev_data[k + stride - 1] = dev_data[k + half_stride - 1] + dev_data[k + stride - 1];
        }

        void scan_gpu_upsweep(int n, int* dev_data) {
            int d_max = ilog2(n) - 1;
            int num_threads = n / 2;
            for (int d = 0; d <= d_max; d++) {
                int numBlocks = 0;
                int blockSize = 0;
                pick_block_size(kernUpsweep, num_threads, &numBlocks, &blockSize);

                kernUpsweep << <numBlocks, blockSize >> > (n, num_threads, d, dev_data);

                num_threads /= 2;
            }
        }

        __global__ void kernDownsweep(int data_length, int active_thread_count, int d, int* dev_data) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= active_thread_count) {
                return;
            }

            int half_stride = 1 << d;
            int stride = 1 << (d + 1);

            int k = index * stride;

            if (k + stride - 1 >= data_length) {
                return;
            }

            int t = dev_data[k + half_stride - 1];
            dev_data[k + half_stride - 1] = dev_data[k + stride - 1];
            dev_data[k + stride - 1] = t + dev_data[k + stride - 1];
        }

        void scan_gpu_downsweep(int n, int* dev_data) {
            cudaMemset(dev_data + (n - 1), 0, sizeof(int));

            int d_max = ilog2ceil(n) - 1;
            int num_threads = 1;
            for (int d = d_max; d >= 0; d--) {
                int numBlocks = 0;
                int blockSize = 0;
                pick_block_size(kernDownsweep, num_threads, &numBlocks, &blockSize);

                kernDownsweep << <numBlocks, blockSize >> > (n, num_threads, d, dev_data);

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
            if (n == 0) {
                timer().endGpuTimer();
                return 0;
            }

            int pow_of_two = ilog2ceil(n);
            int n_new = ipow2(pow_of_two);

            // Allocate buffers
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n_new * sizeof(int));
            cudaMalloc((void**)&dev_indices, n_new * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // Copy input to CPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Convert to bools (only first n elements is necessary because of the Memset)
            map_to_boolean_gpu(n, dev_bools, dev_idata);

            // Initialize indices with 0s so unset inputs are not counted in the scan
            cudaMemset(dev_indices, 0, n_new * sizeof(int));
            // Copy bools to indices since scan operates in place
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            // Scan to get indices
            scan_gpu(n_new, dev_indices);

            // Scatter first n elements
            scatter_gpu(n, dev_odata, dev_idata, dev_bools, dev_indices);

            // Copy output data to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Copy the last index in the index buffer to host
            int last_index = 0;
            cudaMemcpy(&last_index, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            int last_value = idata[n - 1];

            int count_remaining = last_index + (last_value != 0);

            // Free buffers
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);
            cudaFree(dev_idata);
            timer().endGpuTimer();
            return count_remaining;
        }

        void map_to_boolean_gpu(int n, int* dev_bools, const int* dev_idata) {
            int numBlocks = 0;
            int blockSize = 0;
            pick_block_size(Common::kernMapToBoolean, n, &numBlocks, &blockSize);

            Common::kernMapToBoolean << <numBlocks, blockSize >> > (n, dev_bools, dev_idata);
        }

        void scatter_gpu(int n, int* dev_odata,
            const int* dev_idata, const int* dev_bools, const int* dev_indices) {
            int numBlocks = 0;
            int blockSize = 0;
            pick_block_size(Common::kernScatter, n, &numBlocks, &blockSize);

            Common::kernScatter << <numBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
        }
    }
}
