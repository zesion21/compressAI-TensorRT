#include "iostream"
#include "string"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <cstring>
#include "vector"
#include "fstream"
#include <array>

#include "image.h"
#include "encode.h"
#include "read_raw_cuda.h"

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

Logger gLogger;

struct CompressResult
{
    std::array<int, 2> shape;
    std::vector<std::string> strings;

    CompressResult(const std::vector<std::string> &s, const std::array<int, 2> &sh)
        : shape(sh), strings(s) {}
};

class CompressBase
{
private:
    std::string model_name_;
    int quality_;

    void writeTxt(size_t size, auto indexes, std::string name);
    void readJson(const std::string jsonPath)
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

        N = j["N"].get<int>();

        M = j["M"].get<int>();

        shape_scale = j["shape_scale"].get<int>();
    }
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
        // int nbTensors = engine->getNbIOTensors();
        // std::cout << "Tensors Count: " << nbTensors << std::endl;

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

protected:
    std::vector<std::vector<int32_t>> cdfs;
    std::vector<int32_t> cdf_lengths;
    std::vector<int32_t> offsets;
    nvinfer1::ICudaEngine *engine;
    int M;
    int N;
    int shape_scale;

public:
    CompressBase(std::string model_name, int quality)
    {
        model_name_ = model_name;
        quality_ = quality;

        std::string engine_file = "../../models/" + model_name + "_" + std::to_string(quality) + ".trt";
        engine = loadEngine(engine_file);
        if (!engine)
        {
            throw std::runtime_error("Failed to load engine " + engine_file);
        }

        std::string json_path = "../../models/cdf_" + model_name + "_" + std::to_string(quality) + ".json";
        readJson(json_path);
    }
    virtual CompressResult compress(std::vector<float> input_data, nlohmann::json &j) = 0;

    int readRawGPU(std::ifstream *file, std::vector<float> &output, int &orig_h, int &orig_w, int preHeight, int &max_val, int &min_val);

    int compressImage(std::string image_path, std::string out_path, int patch_size);

    int compressRaw(std::string input_path, std::string out_path, int patch_size);
};

// 用来输出张量到csv
void CompressBase::writeTxt(size_t size, auto indexes, std::string name)
{
    std::ofstream outfile("../../out/entropy_cdf_" + name + ".csv"); // 打开或创建文件
    if (!outfile.is_open())
    {
        std::cerr << "无法打开文件!" << std::endl;
        return;
    }

    bool w = false;
    for (int i = 0; i < size; i++)
    {
        if (i > 30000)
            break;

        if (indexes[i] > 0)
            w = true;

        if (w)
            outfile << indexes[i] << "\n";
    }
    outfile.close(); // 关闭文件
}

int CompressBase::readRawGPU(std::ifstream *file, std::vector<float> &output, int &orig_h, int &orig_w, int preHeight, int &max_val, int &min_val)
{

    int xReadSize = 9052;
    orig_w = xReadSize;

    int column = 11936;
    int size = preHeight * column * 2; // 每块的大小

    if (!file->is_open())
    {
        throw std::runtime_error("Raw File is not opend");
    }

    char *buffer = new char[size];
    file->read(buffer, size);
    int readSize = static_cast<int>(file->gcount());
    std::cout << "Bytes read: " << readSize << std::endl;
    readRawCUDA(buffer, readSize, output, orig_h, orig_w, preHeight, max_val, min_val);
    return 0;
}

int CompressBase::compressImage(std::string image_path, std::string out_path, int patch_size)
{
    std::vector<std::vector<float>> datas;
    nlohmann::json patchInfo = nlohmann::json::array();

    writeTxt(cdf_lengths.size(), cdf_lengths.data(), "cdf_lengths");

    int orig_h = 0, orig_w = 0;
    split_image(image_path, datas, patchInfo, orig_h, orig_w, patch_size);

    std::ofstream outFileBin(out_path + ".bin", std::ios::binary);
    if (!outFileBin)
    {
        std::cerr << "Error opening file for writing!" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < patchInfo.size(); i++)
    {
        CompressResult res = compress(datas[i], patchInfo[i]);
        std::string data = res.strings[0];

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
    jOut["model_name"] = model_name_;
    jOut["quality"] = quality_;

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

int CompressBase::compressRaw(std::string input_path, std::string out_path, int patch_size)
{

    int rawPatch = 0; // 拆分大图的 景编号

    int preHeight = 9344;
    int preSize = 1 * 3 * 9052 * preHeight;

    std::ifstream file(input_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Can't open : " + input_path);
    }

    file.seekg(preHeight * 23872, std::ios::cur);

    while (true)
    {

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "start to handle part " << rawPatch << std::endl;

        std::string out_path_i = out_path + ("_" + std::to_string(rawPatch));
        std::vector<float> datas;
        int orig_h = 0, orig_w = 0;
        int max_val = 0, min_val = 65535;

        readRawGPU(&file, datas, orig_h, orig_w, preHeight, max_val, min_val);

        std::cout << "input patch size :" << datas.size() << std::endl;

        nlohmann::json patchInfo = nlohmann::json::array();
        std::ofstream outFileBin(out_path_i + ".bin", std::ios::binary);

        if (!outFileBin)
        {
            std::cerr << "Error opening file for writing!" << std::endl;
            return 1;
        }

        int y = 0;
        while (y < orig_h)
        {
            int endY = MIN(y + patch_size, orig_h);
            int x = 0;
            while (x < orig_w)
            {
                int endX = MIN(x + patch_size, orig_w);
                nlohmann::json j;
                j["y"] = y;
                j["x"] = x;
                j["height"] = (endY - y);
                j["width"] = (endX - x);
                std::vector<float> d;
                getRawPart(datas, d, x, endX, y, endY, orig_h, orig_w);
                CompressResult res = compress(d, j);
                std::string str = res.strings[0];
                outFileBin.write(str.c_str(), str.size());
                patchInfo.push_back(j);
                x = x + patch_size;
            }
            y = y + patch_size;
        };

        if (!outFileBin)
        {
            std::cerr << "Error writing to file!" << std::endl;
            return 1;
        }

        outFileBin.close();
        std::cout << "String written to binary file: " << out_path_i + ".bin" << " successfully!" << std::endl;

        nlohmann::json jOut;
        jOut["original_height"] = orig_h;
        jOut["original_width"] = orig_w;
        jOut["patch_size"] = patch_size;
        jOut["patches"] = patchInfo;
        jOut["max_val"] = max_val;
        jOut["min_val"] = min_val;
        jOut["model_name"] = model_name_;
        jOut["quality"] = quality_;

        // 导出到文件
        std::ofstream out_file_json(out_path_i + ".json");
        if (!out_file_json.is_open())
        {
            std::cerr << "can not write json" << std::endl;
        }

        // 写入格式化的 JSON（2空格缩进）
        out_file_json << jOut.dump(2);
        out_file_json.close();
        std::cout << "Json written to file: " << out_path_i + ".json" << " successfully!" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "part " << rawPatch << " is over! Cast:" << dur << std::endl;

        if (datas.size() < preSize)
            break;
        else
            rawPatch++;
    }

    return 0;
}
