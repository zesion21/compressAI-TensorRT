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

int *compute_padding(int h, int w, int min_div)
{

    int out_h = (h + min_div - 1) / min_div * min_div;
    int out_w = (w + min_div - 1) / min_div * min_div;

    int left = (out_w - w) / 2;     // left
    int right = (out_w - w) - left; // right

    int top = (out_h - h) / 2;      // top
    int bottom = (out_h - h) - top; // bottom

    return new int[4]{left, right, top, bottom};
}

void fill_pading(std::vector<float> &tensor, int &height, int &width, int *pad)
{

    int left = pad[0], right = pad[1], top = pad[2], bottom = pad[3];

    // 填充后的高度和宽度
    int height_add = height + top + bottom;
    int width_add = width + left + right;

    std::vector<float> data(1 * 3 * height_add * width_add);

    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < height_add; ++h)
        {
            for (int w = 0; w < width_add; ++w)
            {
                if (w < left || h < top || w >= width_add - right || h >= height_add - bottom)
                    data[c * height_add * width_add + h * width_add + w] = 0.00;
                else
                    data[c * height_add * width_add + h * width_add + w] =
                        tensor[c * height * width + (h - top) * width + (w - left)];
            }
        }
    }

    tensor = data;
    height = height_add;
    width = width_add;
}