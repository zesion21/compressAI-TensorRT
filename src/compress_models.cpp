#include "../includes/compress_models.h"
#include "vector"

bmshj2018Factorized::bmshj2018Factorized(int quality) : CompressBase("bmshj2018-factorized", quality)
{
    quality_ = quality;
}
CompressResult bmshj2018Factorized::compress(float *input, int height, int width)
{

    int latent_h = height / 16; // 假设 4 次 stride=2 降采样
    int latent_w = width / 16;
    std::cout << "shape:[" << latent_h << "," << latent_w << "]" << std::endl;
    std::vector<float> x_out(1 * 320 * latent_w * latent_h);
    std::vector<int32_t> indexes(1 * 320 * latent_w * latent_h);
    std::vector<float> medians(1 * 320 * 1 * 1);

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape("input", nvinfer1::Dims4{1, 3, height, width});

    // 分配缓冲区
    void *buffers[4];
    cudaMalloc(&buffers[0], 1 * 3 * height * width * sizeof(float));        // input
    cudaMalloc(&buffers[1], 1 * 320 * latent_h * latent_w * sizeof(float)); // x_out
    cudaMalloc(&buffers[2], 1 * 320 * latent_h * latent_w * sizeof(float)); // indexes
    cudaMalloc(&buffers[3], 1 * 320 * 1 * 1 * sizeof(float));               // means

    // 拷贝输入到 GPU
    cudaMemcpy(buffers[0], input, 1 * 3 * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // 执行推理
    context->executeV2(buffers);

    // 拷贝输出到 CPU
    cudaMemcpy(x_out.data(), buffers[1], 1 * 320 * latent_h * latent_w * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(indexes.data(), buffers[2], 1 * 320 * latent_h * latent_w * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(medians.data(), buffers[3], 1 * 320 * 1 * 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理
    for (int i = 0; i < 4; ++i)
        cudaFree(buffers[i]);
    // context->
    delete context;

    std::vector<int32_t> out_put(1 * 320 * latent_w * latent_h);
    quantize_by_block(x_out, medians, out_put, 1, 320, latent_h, latent_w);

    RansEncoder encoder;
    std::string str = encoder.encode_with_indexes(out_put, indexes, cdfs, cdf_lengths, offsets);

    std::vector<std::string> strings = {str};

    CompressResult result(strings, {latent_h, latent_w});

    return result;
}
