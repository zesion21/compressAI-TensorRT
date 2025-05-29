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

#include "includes/compress_models.hpp"

int compressImageTest()
{
    auto start = std::chrono::high_resolution_clock::now();
    // CompressEngine compressEngine("bmshj2018_factorized", 8);

    std::string imagePath = "../../out/1.jpg";
    std::string out_path = "../../out/compress";
    int patch_size = 3200;

    bmshj2018Factorized model(8);
    model.compressImage(imagePath, out_path, patch_size);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "exit with message success! Cast:" << dur.count() << "ms." << std::endl;
    return 0;
}

int compressRaw()
{

    auto start = std::chrono::high_resolution_clock::now();
    // CompressEngine compressEngine("bmshj2018_factorized", 8);

    std::string input_path = "F:/compressAI/test_data/decrypt1.raw";
    std::string out_path = "../../out/raw_compress/decrypt1";

    bmshj2018Factorized model(8);
    model.compressRaw(input_path, out_path, 3200);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "exit with code 0 ! Cast " << dur << "." << std::endl;

    return 0;
}

int main()
{

    compressImageTest();
    return 0;
}