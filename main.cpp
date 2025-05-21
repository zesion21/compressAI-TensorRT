#include <NvInfer.h>          // TensorRT 核心头文件
#include <cuda_runtime_api.h> // CUDA 运行时
#include <fstream>            // 文件操作
#include <vector>             // 数据存储
#include <iostream>           // 调试输出
#include <memory>             // std::unique_ptr
#include <string>             // std::string
#include <stdexcept>          // std::runtime_error
#include <cstring>            // std::strcmp
#include <map>
#include "includes/image.h"
#include "includes/encode.h"
#include "includes/rans_interface.hpp"

// 自定义 TensorRT 日志类
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // 仅打印 ERROR 和 WARNING 级别的日志
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[" << static_cast<int>(severity) << "] " << msg << std::endl;
        }
    }
}; // 确保分号

// 全局 Logger 对象
Logger gLogger;

// // 检查 CUDA 错误的宏
// #define CHECK_CUDA(call)                                                                                         \
//     do                                                                                                           \
//     {                                                                                                            \
//         cudaError_t err = call;                                                                                  \
//         if (err != cudaSuccess)                                                                                  \
//         {                                                                                                        \
//             std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << " 在第 " << __LINE__ << " 行" << std::endl; \
//             std::exit(1);                                                                                        \
//         }                                                                                                        \
//     } while (0)

// // 检查 TensorRT 指针的宏
// #define CHECK_PTR(ptr, msg)                                                             \
//     do                                                                                  \
//     {                                                                                   \
//         if (!ptr)                                                                       \
//         {                                                                               \
//             std::cerr << "错误: " << msg << " 在第 " << __LINE__ << " 行" << std::endl; \
//             std::exit(1);                                                               \
//         }                                                                               \
//     } while (0)

// 加载 Engine
nvinfer1::ICudaEngine *loadEngine(const std::string &engineFile)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
        return nullptr;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    return runtime->deserializeCudaEngine(engineData.data(), size);
}

// 推理函数
void infer(nvinfer1::ICudaEngine *engine, float *input, int height, int width, float *x_out, int32_t *indexes, float *medians)
{
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    // 设置输入形状
    int latent_h = height / 16; // 假设 4 次 stride=2 降采样
    int latent_w = width / 16;
    context->setInputShape("input", nvinfer1::Dims4{1, 3, height, width});

    // 分配缓冲区
    void *buffers[4];
    cudaMalloc(&buffers[0], 1 * 3 * height * width * sizeof(float));        // input
    cudaMalloc(&buffers[1], 1 * 320 * latent_h * latent_w * sizeof(float)); // x_out
    cudaMalloc(&buffers[2], 1 * 320 * latent_h * latent_w * sizeof(float)); // indexes
    cudaMalloc(&buffers[3], 1 * 320 * 1 * 1 * sizeof(float));               // means

    // 拷贝输入到 GPU
    cudaMemcpy(buffers[0], input, 1 * 3 * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // 执行推理
    context->executeV2(buffers);

    // 拷贝输出到 CPU
    cudaMemcpy(x_out, buffers[1], 1 * 320 * latent_h * latent_w * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(indexes, buffers[2], 1 * 320 * latent_h * latent_w * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(medians, buffers[3], 1 * 320 * 1 * 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理
    for (int i = 0; i < 4; ++i)
        cudaFree(buffers[i]);
    // context->destroy();
}

// 用来输出张量到csv
void writeTxt(std::vector<int32_t> indexes, std::string name)
{

    std::ofstream outfile("../../out/entropy_cdf_" + name + ".csv"); // 打开或创建文件
    if (!outfile.is_open())
    {
        std::cerr << "无法打开文件!" << std::endl;
        return;
    }

    for (size_t i = 0; i < indexes.size(); i++)
    {
        // int32_t indexes_put = static_cast<int32_t>(round(indexes[i]));
        outfile << indexes[i] << "\n";
    }

    // outfile << "Hello, world!\n";
    // outfile << "写入一行文字到文本文件中。\n";
    outfile.close(); // 关闭文件（最好手动关闭）
}

void writeTxt(std::vector<float> indexes, std::string name)
{

    std::ofstream outfile("../../out/entropy_cdf_" + name + ".csv"); // 打开或创建文件
    if (!outfile.is_open())
    {
        std::cerr << "无法打开文件!" << std::endl;
        return;
    }

    for (size_t i = 0; i < indexes.size(); i++)
    {
        // int32_t indexes_put = static_cast<int32_t>(round(indexes[i]));
        outfile << indexes[i] << "\n";
    }

    // outfile << "Hello, world!\n";
    // outfile << "写入一行文字到文本文件中。\n";
    outfile.close(); // 关闭文件（最好手动关闭）
}

// 生成bin文件
int writeFile(std::string data, std::string outPath)
{
    std::ofstream outFile(outPath, std::ios::binary);

    if (!outFile)
    {
        std::cerr << "Error opening file for writing!" << std::endl;
        return 1;
    }

    size_t length = data.size();
    outFile.write(data.c_str(), length);

    if (!outFile)
    {
        std::cerr << "Error writing to file!" << std::endl;
        return 1;
    }

    // 关闭文件
    outFile.close();
    std::cout << "String written to binary file successfully!" << std::endl;
    return 0;
}

int main()
{

    auto start = std::chrono::high_resolution_clock::now();

    int imageHeight = 0;
    int imageWidth = 0;

    std::string imagePath = "E:/work2/pythonProjects/CompressAI-master/examples/assets/stmalo_fracape.png";
    // std::string imagePath = "E:/work2/c/engine_runtime/1.jpg";
    const std::string engine_file = "../models/encoder_max_2048.trt";

    // 0.将图片转化为张量
    std::vector<float> input_data = imageToTensor(imagePath, imageHeight, imageWidth);
    std::cout << input_data.size() << std::endl;

    auto engine = loadEngine(engine_file);
    if (!engine)
    {
        std::cerr << "Failed to load engine" << std::endl;
        return -1;
    }

    int nbTensors = engine->getNbIOTensors();
    std::cout << "Tensors Count: " << nbTensors << std::endl;

    for (int i = 0; i < nbTensors; ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        auto shape = engine->getTensorShape(tensor_name);
        int size = 1;
        std::cout << "Tensor " << i << ": " << tensor_name << " (";
        std::cout << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "input" : io_mode == nvinfer1::TensorIOMode::kOUTPUT ? "output"
                                                                                                                       : "other");
        std::cout << "), shape: [";
        for (int j = 0; j < shape.nbDims; ++j)
        {
            size *= shape.d[j];
            std::cout << shape.d[j] << (j < shape.nbDims - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // 分配缓冲区
    }

    int latent_h = imageHeight / 16;
    int latent_w = imageWidth / 16;
    std::vector<float> x_out(1 * 320 * latent_w * latent_h);
    std::vector<int32_t> indexes(1 * 320 * latent_w * latent_h);
    std::vector<float> medians(1 * 320 * 1 * 1);

    infer(engine, input_data.data(), imageHeight, imageWidth, x_out.data(), indexes.data(), medians.data());

    std::vector<int32_t> out_put(1 * 320 * latent_w * latent_h);
    std::vector<int32_t> indexes_put(1 * 320 * latent_w * latent_h);

    // writeTxt(input_data, "input_data");
    // writeTxt(indexes, "indexes");
    // writeTxt(x_out, "x_out");
    // writeTxt(medians, "medians");

    quantize_by_block(x_out, medians, out_put, 1, 320, latent_h, latent_w);

    RansEncoder encoder;

    std::vector<std::vector<int32_t>> cdfs;
    std::vector<int32_t> cdf_lengths;
    std::vector<int32_t> offsets;
    readJson(cdfs, cdf_lengths, offsets);
    std::string str = encoder.encode_with_indexes(out_put, indexes, cdfs, cdf_lengths, offsets);
    std::cout << "encode over" << std::endl;

    writeFile(str, "../../out/1_" + std::to_string(imageWidth) + "_" + std::to_string(imageHeight) + ".bin");

    auto end = std::chrono::high_resolution_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "exit with message success! Cast:" << dur.count() << "ms. medians  Size:" << medians.size()
              << " x_out  Size:" << x_out.size()
              << " indexes  Size:" << indexes.size()
              << std::endl;

    return 0;
}