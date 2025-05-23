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
#include "includes/json.hpp"

// 自定义 TensorRT 日志类
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[" << static_cast<int>(severity) << "] " << msg << std::endl;
        }
    }
};

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

class CompressEngine
{
private:
    nvinfer1::ICudaEngine *engine;

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
        nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineData.data(), size);

        //  打印张量
        //  int nbTensors = engine->getNbIOTensors();
        //  std::cout << "Tensors Count: " << nbTensors << std::endl;

        // for (int i = 0; i < nbTensors; ++i)
        // {
        //     const char *tensor_name = engine->getIOTensorName(i);
        //     nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        //     auto shape = engine->getTensorShape(tensor_name);

        //     std::cout << "Tensor " << i << ": " << tensor_name << " (";
        //     std::cout << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "input" : io_mode == nvinfer1::TensorIOMode::kOUTPUT ? "output"
        //                                                                                                                    : "other");
        //     std::cout << "), shape: [";
        //     for (int j = 0; j < shape.nbDims; ++j)
        //     {
        //         std::cout << shape.d[j] << (j < shape.nbDims - 1 ? ", " : "");
        //     }
        //     std::cout << "]" << std::endl;
        // }

        return engine;
    }

    // 推理函数
    void infer(float *input, int height, int width, float *x_out, int32_t *indexes, float *medians)
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
        // context->
        delete context;
    }

    // 用来输出张量到csv
    void writeTxt(int size, auto indexes, std::string name)
    {
        std::ofstream outfile("../../out/entropy_cdf_" + name + ".csv"); // 打开或创建文件
        if (!outfile.is_open())
        {
            std::cerr << "无法打开文件!" << std::endl;
            return;
        }

        for (size_t i = 0; i < size; i++)
        {
            outfile << indexes[i] << "\n";
        }
        outfile.close(); // 关闭文件
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

        outFile.close();
        std::cout << "String written to binary file successfully!" << std::endl;
        return 0;
    }

public:
    CompressEngine(std::string enginePath)
    {
        engine = loadEngine(enginePath);
        if (!engine)
        {
            throw std::runtime_error("Failed to load engine " + enginePath);
        }
    }

    ~CompressEngine() {};
    std::string compress(std::vector<float> input_data, nlohmann::json &j)
    {

        int imageHeight = j["height"].get<int>(), imageWidth = j["width"].get<int>();
        // 0.将图片转化为张量
        // std::vector<float> input_data = imageToTensor(input_path, imageHeight, imageWidth);
        std::cout << "input size : " << input_data.size() << std::endl;
        int *pad = compute_padding(imageHeight, imageWidth, 64);

        // 填充标准
        int outHeight = imageHeight, outWidth = imageWidth;
        fill_pading(input_data, outHeight, outWidth, pad);

        std::cout << "standard image shape:[" << outHeight << "," << outWidth << "]" << std::endl;

        int latent_h = outHeight / 16;
        int latent_w = outWidth / 16;
        std::cout << "shape:[" << latent_h << "," << latent_w << "]" << std::endl;
        std::vector<float> x_out(1 * 320 * latent_w * latent_h);
        std::vector<int32_t> indexes(1 * 320 * latent_w * latent_h);
        std::vector<float> medians(1 * 320 * 1 * 1);

        infer(input_data.data(), outHeight, outWidth, x_out.data(), indexes.data(), medians.data());

        std::vector<int32_t> out_put(1 * 320 * latent_w * latent_h);
        std::vector<int32_t> indexes_put(1 * 320 * latent_w * latent_h);

        // writeTxt(input_data.size(), input_data.data(), "input_data");
        // writeTxt(indexes, "indexes");
        // writeTxt(x_out, "x_out");
        // writeTxt(medians, "medians");

        quantize_by_block(x_out, medians, out_put, 1, 320, latent_h, latent_w);

        std::vector<std::vector<int32_t>> cdfs;
        std::vector<int32_t> cdf_lengths;
        std::vector<int32_t> offsets;
        readJson(cdfs, cdf_lengths, offsets);

        RansEncoder encoder;
        std::string str = encoder.encode_with_indexes(out_put, indexes, cdfs, cdf_lengths, offsets);

        j["shape"] = {latent_h, latent_w};
        j["string_lengths"] = {str.size()};

        return str;

        // writeFile(str, out_path + "_" + std::to_string(imageWidth) + "_" + std::to_string(imageHeight) + ".bin");
        // return 0;
    }

    int compressImage(std::string image_path, std::string out_path, int patch_size)
    {

        std::vector<std::vector<float>> datas;
        nlohmann::json patchInfo = nlohmann::json::array();
        std::ofstream outFileBin(out_path + ".bin", std::ios::binary);

        if (!outFileBin)
        {
            std::cerr << "Error opening file for writing!" << std::endl;
            return 1;
        }

        int orig_h = 0, orig_w = 0;
        split_image(image_path, datas, patchInfo, orig_h, orig_w, patch_size);

        for (size_t i = 0; i < patchInfo.size(); i++)
        {
            std::string data = compress(datas[i], patchInfo[i]);
            size_t length = data.size();
            outFileBin.write(data.c_str(), length);
        }

        if (!outFileBin)
        {
            std::cerr << "Error writing to file!" << std::endl;
            return 1;
        }

        outFileBin.close();
        std::cout << "String written to binary file successfully!" << std::endl;

        nlohmann::json jOut;
        jOut["original_height"] = orig_h;
        jOut["original_width"] = orig_w;
        jOut["patch_size"] = patch_size;
        jOut["patches"] = patchInfo;
        // 导出到文件
        std::ofstream out_file_json(out_path + ".json");
        if (!out_file_json.is_open())
        {
            std::cerr << "can not write json" << std::endl;
        }

        // 写入格式化的 JSON（2空格缩进）
        out_file_json << jOut.dump(2);
        out_file_json.close();
        std::cout << "Json written to file successfully!" << std::endl;

        return 0;
    }
};

int compressImageTest()
{
    auto start = std::chrono::high_resolution_clock::now();
    const std::string engine_file = "../models/encoder_3200.trt";
    CompressEngine compressEngine(engine_file);

    std::string imagePath = "../../out/shiyan.png";
    std::string out_path = "../../out/compress";
    int patch_size = 3200;

    compressEngine.compressImage(imagePath, out_path, patch_size);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "exit with message success! Cast:" << dur.count() << "ms." << std::endl;
    return 0;
}

int main()
{

    compressImageTest();
    return 0;
}