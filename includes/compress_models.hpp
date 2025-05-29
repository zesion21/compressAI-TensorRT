
#include "rans_interface.hpp"
#include "compress_base.hpp"

class bmshj2018Factorized : public CompressBase
{
private:
    void infer(float *input, int height, int width, float *x_out, int32_t *indexes, float *medians)
    {
        nvinfer1::IExecutionContext *context(engine->createExecutionContext());
        // 设置输入形状
        int latent_h = height / shape_scale; // 假设 4 次 stride=2 降采样
        int latent_w = width / shape_scale;
        context->setInputShape("input", nvinfer1::Dims4{1, 3, height, width});
        // 分配缓冲区
        void *buffers[4];
        cudaMalloc(&buffers[0], 1 * 3 * height * width * sizeof(float));      // input
        cudaMalloc(&buffers[1], 1 * M * latent_h * latent_w * sizeof(float)); // x_out
        cudaMalloc(&buffers[2], 1 * M * latent_h * latent_w * sizeof(float)); // indexes
        cudaMalloc(&buffers[3], 1 * M * 1 * 1 * sizeof(float));               // means

        // 拷贝输入到 GPU
        cudaMemcpy(buffers[0], input, 1 * 3 * height * width * sizeof(float), cudaMemcpyHostToDevice);
        // 执行推理
        context->executeV2(buffers);
        // 拷贝输出到 CPU
        cudaMemcpy(x_out, buffers[1], 1 * M * latent_h * latent_w * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(indexes, buffers[2], 1 * M * latent_h * latent_w * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(medians, buffers[3], 1 * M * 1 * 1 * sizeof(float), cudaMemcpyDeviceToHost);
        // 清理
        for (int i = 0; i < 4; ++i)
            cudaFree(buffers[i]);
        delete context;
    }

public:
    bmshj2018Factorized(int quality) : CompressBase("bmshj2018-factorized", quality)
    {
    }

    CompressResult compress(std::vector<float> input_data, nlohmann::json &j) override
    {

        int imageHeight = j["height"].get<int>(), imageWidth = j["width"].get<int>();

        int *pad = compute_padding(imageHeight, imageWidth, 64);

        // 填充标准
        int outHeight = imageHeight, outWidth = imageWidth;
        fill_pading(input_data, outHeight, outWidth, pad);
        std::cout << "standard image shape:[" << outHeight << "," << outWidth << "]" << std::endl;

        int latent_h = outHeight / shape_scale;
        int latent_w = outWidth / shape_scale;
        std::cout << "shape:[" << latent_h << "," << latent_w << "]" << std::endl;

        std::vector<float> x_out(1 * M * latent_w * latent_h);
        std::vector<int32_t> indexes(1 * M * latent_w * latent_h);
        std::vector<float> medians(1 * M * 1 * 1);

        infer(input_data.data(), outHeight, outWidth, x_out.data(), indexes.data(), medians.data());

        std::vector<int32_t> out_put(1 * M * latent_w * latent_h);
        std::vector<int32_t> indexes_put(1 * M * latent_w * latent_h);

        quantize_by_block(x_out, medians, out_put, 1, M, latent_h, latent_w);

        RansEncoder encoder;
        std::string str = encoder.encode_with_indexes(out_put, indexes, cdfs, cdf_lengths, offsets);

        j["shape"] = {latent_h, latent_w};
        j["string_lengths"] = {str.size()};

        std::vector<std::string> strings;
        strings.clear();
        strings.reserve(1);
        strings.push_back(str);
        CompressResult res(strings, {latent_h, latent_w});
        return res;
    }
};
