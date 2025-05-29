#include "iostream"
#include "vector"
#include "image.h"
#include "fstream"
#include "NvInfer.h"
#include "read_raw_cuda.h"
#include "encode.h"
#include "rans_interface.hpp"
#include "compress_base.h"
#include <array>

struct CompressResult
{
    std::array<int, 2> shape;
    std::vector<std::string> strings;

    CompressResult(const std::vector<std::string> &s, const std::array<int, 2> &sh)
        : shape(sh), strings(s) {}
};

class bmshj2018Factorized : public CompressBase
{
private:
    int quality_;
    /* data */
public:
    bmshj2018Factorized(int quality);

    CompressResult compress(float *input, int height, int width);
};
