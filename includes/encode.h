#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <cstdint> // for int32_t
#include "json.hpp"

void quantize_by_block(
    const std::vector<float> &inputs,
    const std::vector<float> &means,
    std::vector<int32_t> &outputs,
    int N, int C, int H, int W);

void readJson(std::vector<std::vector<int32_t>> &cdfs, std::vector<int32_t> &cdf_lengths, std::vector<int32_t> &offsets);

int *compute_padding(int h, int w, int min_div);

// 填充张量
void fill_pading(std::vector<float> &tensor, int &height, int &width, int *pad);
