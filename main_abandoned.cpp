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

// 检查 CUDA 错误的宏
#define CHECK_CUDA(call)                                                                                         \
    do                                                                                                           \
    {                                                                                                            \
        cudaError_t err = call;                                                                                  \
        if (err != cudaSuccess)                                                                                  \
        {                                                                                                        \
            std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << " 在第 " << __LINE__ << " 行" << std::endl; \
            std::exit(1);                                                                                        \
        }                                                                                                        \
    } while (0)

// 检查 TensorRT 指针的宏
#define CHECK_PTR(ptr, msg)                                                             \
    do                                                                                  \
    {                                                                                   \
        if (!ptr)                                                                       \
        {                                                                               \
            std::cerr << "错误: " << msg << " 在第 " << __LINE__ << " 行" << std::endl; \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

int main()
{

    int imageHeight = 0;
    int imageWidth = 0;
    std::string imagePath = "E:/work2/pythonProjects/CompressAI-master/examples/assets/stmalo_fracape.png";

    // 0.将图片转化为张量
    std::vector<float> tensor = imageToTensor(imagePath, imageHeight, imageWidth);

    std::cout << tensor.size() << std::endl;

    // --- 1. 定义模型参数 ---
    const std::string engine_file = "E:\\work2\\pythonProjects\\CompressAI-master\\examples\\bmshj2018_factorized_encoder_2.engine";
    // const std::string engine_file = "E:/work2/c/engine_runtime/bmshj2018_factorized_encoder.trt";
    const char *input_name = "input";
    const char *z_hat_name = "z_hat";
    const char *likelihoods_name = "likelihoods";

    // --- 2. 读取 TensorRT Engine 文件 ---
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "无法打开 Engine 文件: " << engine_file << std::endl;
        return 1;
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    if (buffer.empty())
    {
        std::cerr << "Engine 文件为空" << std::endl;
        return 1;
    }

    // --- 3. 创建 TensorRT Runtime ---
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    CHECK_PTR(runtime, "创建 TensorRT Runtime 失败");

    // --- 4. 反序列化 Engine ---
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    CHECK_PTR(engine, "反序列化 Engine 失败");

    // --- 5. 创建执行上下文 ---
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    CHECK_PTR(context, "创建执行上下文失败");

    // --- 6. 获取张量信息 ---
    int nbTensors = engine->getNbIOTensors();
    std::cout << "张量数量: " << nbTensors << std::endl;
    std::map<std::string, void *> tensor_buffers;
    std::map<std::string, int> tensor_sizes;
    std::map<std::string, int> tensor_indices;        // 张量名称到绑定索引的映射
    std::vector<void *> bindings(nbTensors, nullptr); // 绑定数组

    bool found_input = false, found_z_hat = false, found_likelihoods = false;
    for (int i = 0; i < nbTensors; ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        auto shape = engine->getTensorShape(tensor_name);
        int size = 1;
        std::cout << "张量 " << i << ": " << tensor_name << " (";
        std::cout << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "输入" : io_mode == nvinfer1::TensorIOMode::kOUTPUT ? "输出"
                                                                                                                      : "其他");
        std::cout << "), 形状: [";
        for (int j = 0; j < shape.nbDims; ++j)
        {
            size *= shape.d[j];
            std::cout << shape.d[j] << (j < shape.nbDims - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // 分配缓冲区
        void *buffer = nullptr;
        CHECK_CUDA(cudaMalloc(&buffer, size * imageHeight * imageWidth * sizeof(float)));
        tensor_buffers[tensor_name] = buffer;
        tensor_sizes[tensor_name] = size * imageHeight * imageWidth;
        tensor_indices[tensor_name] = i; // 记录张量索引
        bindings[i] = buffer;            // 设置绑定数组

        // 验证预期张量
        if (std::strcmp(tensor_name, input_name) == 0)
        {
            found_input = true;
            if (io_mode != nvinfer1::TensorIOMode::kINPUT)
            {
                std::cerr << "错误: " << tensor_name << " 不是输入张量" << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(tensor_name, z_hat_name) == 0)
        {
            found_z_hat = true;
            if (io_mode != nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cerr << "错误: " << tensor_name << " 不是输出张量" << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(tensor_name, likelihoods_name) == 0)
        {
            found_likelihoods = true;
            if (io_mode != nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cerr << "错误: " << tensor_name << " 不是输出张量" << std::endl;
                return 1;
            }
        }
    }

    // 验证所有预期张量
    if (!found_input || !found_z_hat || !found_likelihoods)
    {
        std::cerr << "错误: 未找到所有预期张量: "
                  << "input=" << (found_input ? "找到" : "未找到") << ", "
                  << "z_hat=" << (found_z_hat ? "找到" : "未找到") << ", "
                  << "likelihoods=" << (found_likelihoods ? "找到" : "未找到") << std::endl;
        std::cerr << "请运行 trtexec --verbose 检查张量名称" << std::endl;
        for (auto &buf : tensor_buffers)
        {
            CHECK_CUDA(cudaFree(buf.second));
        }
        return 1;
    }

    // --- 7. 准备输入数据 ---
    // int input_size = tensor_sizes[input_name];
    // std::vector<float> input_data(input_size, 1.0f); // 示例数据
    // CHECK_CUDA(cudaMemcpy(tensor_buffers[input_name], tensor.data(), 1 * 3 * imageHeight * imageWidth * sizeof(float), cudaMemcpyHostToDevice));

    // // --- 8. 设置张量地址 ---
    // for (const auto &buf : tensor_buffers)
    // {
    //     if (!context->setTensorAddress(buf.first.c_str(), buf.second))
    //     {
    //         std::cerr << "设置张量地址失败: " << buf.first << std::endl;
    //         for (auto &b : tensor_buffers)
    //         {
    //             CHECK_CUDA(cudaFree(b.second));
    //         }
    //         return 1;
    //     }
    // }

    // // --- 9. 执行推理 ---
    // if (!context->executeV2(bindings.data()))
    // {
    //     std::cerr << "推理执行失败" << std::endl;
    //     for (auto &buf : tensor_buffers)
    //     {
    //         CHECK_CUDA(cudaFree(buf.second));
    //     }
    //     return 1;
    // }

    // // --- 10. 获取输出 ---
    // int z_hat_size = tensor_sizes[z_hat_name];
    // int likelihoods_size = tensor_sizes[likelihoods_name];
    // std::vector<float> z_hat_data(z_hat_size);

    // std::vector<float> likelihoods_data(likelihoods_size);
    // try
    // {
    //     CHECK_CUDA(cudaMemcpy(z_hat_data.data(), tensor_buffers[z_hat_name], z_hat_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA(cudaMemcpy(likelihoods_data.data(), tensor_buffers[likelihoods_name], likelihoods_size * sizeof(float), cudaMemcpyDeviceToHost));
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "拷贝输出数据失败: " << e.what() << std::endl;
    //     for (auto &buf : tensor_buffers)
    //     {
    //         CHECK_CUDA(cudaFree(buf.second));
    //     }
    //     return 1;
    // }

    // // --- 11. 打印输出 ---
    // std::cout << "推理完成，输出样本:" << std::endl;
    // std::cout << "z_hat[0]: " << z_hat_data[0] << std::endl;
    // std::cout << "likelihoods[0]: " << likelihoods_data[0] << std::endl;

    // // --- 12. 清理 ---
    // for (auto &buf : tensor_buffers)
    // {
    //     CHECK_CUDA(cudaFree(buf.second));
    // }

    std::cout << "程序正常退出" << std::endl;

    return 0;
}