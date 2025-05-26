#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

// 错误检查宏
#define CUDA_CHECK(call)                                                                                \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << " 在行 " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

// 核函数：将字节缓冲区转换为 uint16_t
__global__ void convertToUint16(const char *buffer, uint16_t *raw_data, int readSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 < readSize)
    {
        raw_data[idx] = (static_cast<unsigned char>(buffer[idx * 2]) << 8) | static_cast<unsigned char>(buffer[idx * 2 + 1]);
    }
}

// 核函数：提取指定列
__global__ void extractColumns(const uint16_t *raw_data, uint16_t *tempPANdata, int actual_rows, int xReadSize, int column)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < actual_rows && col < xReadSize)
    {
        tempPANdata[row * xReadSize + col] = raw_data[row * column + 32 + col];
    }
}

// 核函数：归约寻找最小/最大值
__global__ void reduceMinMax(const uint16_t *data, uint16_t *min_out, uint16_t *max_out, int size)
{
    extern __shared__ uint16_t sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    uint16_t min_val = (idx < size) ? data[idx] : 65535;
    uint16_t max_val = (idx < size) ? data[idx] : 0;
    sdata[tid] = min_val;
    sdata[tid + blockDim.x] = max_val;
    __syncthreads();

    // 在共享内存中归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
            sdata[tid + blockDim.x] = max(sdata[tid + blockDim.x], sdata[tid + blockDim.x + s]);
        }
        __syncthreads();
    }

    // 写入该块的结果
    if (tid == 0)
    {
        min_out[blockIdx.x] = sdata[0];
        max_out[blockIdx.x] = sdata[blockDim.x];
    }
}

// 核函数：归一化数据
__global__ void normalizeData(const uint16_t *tempPANdata, float *tempOutput, uint16_t min_val, uint16_t max_val, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float range = static_cast<float>(max_val - min_val);
        if (range == 0)
            range = 1.0f;
        tempOutput[idx] = (tempPANdata[idx] - min_val) / range;
    }
}

// 核函数：复制到三个通道
__global__ void replicateChannels(const float *tempOutput, float *output, int actual_rows, int xReadSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = actual_rows * xReadSize;
    if (idx < total_size)
    {
        output[idx] = tempOutput[idx];
        output[idx + total_size] = tempOutput[idx];
        output[idx + 2 * total_size] = tempOutput[idx];
    }
}

extern "C" int readRawCUDA(char *buffer, int readSize, std::vector<float> &output, int &orig_h, int &orig_w, int preHeight)
{
    int xReadSize = 9052;
    orig_w = xReadSize;
    int column = 11936;
    int size = preHeight * column * 2; // 字节大小

    // 计算实际元素数和行数
    int actual_elements = readSize / 2;
    int actual_rows = actual_elements / column;
    orig_h = actual_rows;
    if (actual_rows == 0)
    {
        output.clear();
        return 0; // 数据不足
    }

    // 设备内存指针
    char *d_buffer;
    uint16_t *d_raw_data;
    uint16_t *d_tempPANdata;
    float *d_tempOutput;
    float *d_output;
    uint16_t *d_min_vals;
    uint16_t *d_max_vals;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_buffer, readSize * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_raw_data, actual_elements * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_tempPANdata, actual_rows * xReadSize * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_tempOutput, actual_rows * xReadSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, actual_rows * xReadSize * 3 * sizeof(float)));

    // 将输入缓冲区复制到设备
    CUDA_CHECK(cudaMemcpy(d_buffer, buffer, readSize * sizeof(char), cudaMemcpyHostToDevice));

    // 核函数 1：转换为 uint16_t
    int threadsPerBlock = 256;
    int blocks = (actual_elements + threadsPerBlock - 1) / threadsPerBlock;
    convertToUint16<<<blocks, threadsPerBlock>>>(d_buffer, d_raw_data, readSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // // 核函数 2：提取列
    dim3 blockDim(32, 32);
    dim3 gridDim((xReadSize + blockDim.x - 1) / blockDim.x, (actual_rows + blockDim.y - 1) / blockDim.y);
    extractColumns<<<gridDim, blockDim>>>(d_raw_data, d_tempPANdata, actual_rows, xReadSize, column);
    CUDA_CHECK(cudaDeviceSynchronize());

    // // 核函数 3：归约寻找最小/最大值
    int data_size = actual_rows * xReadSize;
    blocks = (data_size + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaMalloc(&d_min_vals, blocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_max_vals, blocks * sizeof(uint16_t)));
    reduceMinMax<<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(uint16_t)>>>(d_tempPANdata, d_min_vals, d_max_vals, data_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // // 将最小/最大值结果复制回主机
    std::vector<uint16_t> min_vals(blocks), max_vals(blocks);
    CUDA_CHECK(cudaMemcpy(min_vals.data(), d_min_vals, blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_vals.data(), d_max_vals, blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // // 计算最终最小/最大值
    uint16_t min_val = *std::min_element(min_vals.begin(), min_vals.end());
    uint16_t max_val = *std::max_element(max_vals.begin(), max_vals.end());
    std::cout << "max: " << static_cast<int>(max_val) << " min: " << static_cast<int>(min_val)
              << " range: " << static_cast<float>(max_val - min_val) << std::endl;

    // // 核函数 4：归一化数据
    normalizeData<<<blocks, threadsPerBlock>>>(d_tempPANdata, d_tempOutput, min_val, max_val, data_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // // 核函数 5：复制通道
    replicateChannels<<<blocks, threadsPerBlock>>>(d_tempOutput, d_output, actual_rows, xReadSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // // 将输出复制回主机
    output.clear();
    output.resize(actual_rows * xReadSize * 3);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, actual_rows * xReadSize * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    // // 释放设备内存
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_raw_data));
    CUDA_CHECK(cudaFree(d_tempPANdata));
    CUDA_CHECK(cudaFree(d_tempOutput));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_min_vals));
    CUDA_CHECK(cudaFree(d_max_vals));

    // 释放主机缓冲区
    delete[] buffer;

    return 0;
}