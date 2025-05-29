
#include "../includes/compress_base.h"

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

nvinfer1::ICudaEngine *CompressBase::loadEngine(const std::string &engineFile)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
        return nullptr;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    Logger gLogger;
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

CompressBase::CompressBase(std::string model_name, int q)
{

    std::string pre = "../models/" + model_name + "_" + std::to_string(q);
    std::string enginePath = pre + ".trt";
    engine = loadEngine(enginePath);
    if (!engine)
    {
        throw std::runtime_error("Failed to load engine " + enginePath);
    }
    readJson(pre + ".json");
}

void CompressBase::readJson(const std::string jsonPath)
{

    std::ifstream file(jsonPath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open JSON file.\n";
        // return;
    }
    nlohmann::json j;
    file >> j;

    // 1. 解析 cdf（二维数组）
    cdfs = j["cdf"].get<std::vector<std::vector<int32_t>>>();

    // 2. 解析 cdf_lengths
    cdf_lengths = j["cdf_lengths"].get<std::vector<int32_t>>();

    // 3. 解析 offset
    offsets = j["offset"].get<std::vector<int32_t>>();
}