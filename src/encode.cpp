#include "../includes/encode.h"
#include "iostream"
#include <fstream>

void quantize_by_block(
    const std::vector<float> &inputs,
    const std::vector<float> &means,
    std::vector<int32_t> &outputs,
    int N, int C, int H, int W)
{
    outputs.resize(N * C * H * W);

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            float mean = means[n * C + c]; // means shape: (N, C, 1, 1)
            for (int h = 0; h < H; ++h)
            {
                for (int w = 0; w < W; ++w)
                {
                    int idx = ((n * C + c) * H + h) * W + w;
                    float val = inputs[idx] - mean;
                    outputs[idx] = static_cast<int32_t>(std::round(val));
                }
            }
        }
    }
}

void readJson(std::vector<std::vector<int32_t>> &cdfs, std::vector<int32_t> &cdf_lengths, std::vector<int32_t> &offsets)
{

    std::ifstream file("../models/entropy_cdf.json");
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