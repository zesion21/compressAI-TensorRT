#include "iostream"
#include "string"
#include <cstring>
#include "vector"
#include "NvInfer.h"
#include "fstream"
#include "encode.h"

class CompressBase
{

private:
    nvinfer1::ICudaEngine *loadEngine(const std::string &engineFile);
    void readJson(const std::string jsonPath);

protected:
    nvinfer1::ICudaEngine *engine;
    std::vector<std::vector<int32_t>> cdfs;
    std::vector<int32_t> cdf_lengths;
    std::vector<int32_t> offsets;

public:
    CompressBase(std::string model_name, int q);
};