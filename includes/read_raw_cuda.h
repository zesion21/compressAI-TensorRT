#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

extern "C" int readRawCUDA(char *buffer, int readSize, std::vector<float> &output, int &orig_h, int &orig_w, int preHeight, int &max_val, int &min_val);