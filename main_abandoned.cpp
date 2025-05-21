#include <NvInfer.h>          // TensorRT ����ͷ�ļ�
#include <cuda_runtime_api.h> // CUDA ����ʱ
#include <fstream>            // �ļ�����
#include <vector>             // ���ݴ洢
#include <iostream>           // �������
#include <memory>             // std::unique_ptr
#include <string>             // std::string
#include <stdexcept>          // std::runtime_error
#include <cstring>            // std::strcmp
#include <map>
#include "includes/image.h"

// �Զ��� TensorRT ��־��
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // ����ӡ ERROR �� WARNING �������־
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[" << static_cast<int>(severity) << "] " << msg << std::endl;
        }
    }
}; // ȷ���ֺ�

// ȫ�� Logger ����
Logger gLogger;

// ��� CUDA ����ĺ�
#define CHECK_CUDA(call)                                                                                         \
    do                                                                                                           \
    {                                                                                                            \
        cudaError_t err = call;                                                                                  \
        if (err != cudaSuccess)                                                                                  \
        {                                                                                                        \
            std::cerr << "CUDA ����: " << cudaGetErrorString(err) << " �ڵ� " << __LINE__ << " ��" << std::endl; \
            std::exit(1);                                                                                        \
        }                                                                                                        \
    } while (0)

// ��� TensorRT ָ��ĺ�
#define CHECK_PTR(ptr, msg)                                                             \
    do                                                                                  \
    {                                                                                   \
        if (!ptr)                                                                       \
        {                                                                               \
            std::cerr << "����: " << msg << " �ڵ� " << __LINE__ << " ��" << std::endl; \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

int main()
{

    int imageHeight = 0;
    int imageWidth = 0;
    std::string imagePath = "E:/work2/pythonProjects/CompressAI-master/examples/assets/stmalo_fracape.png";

    // 0.��ͼƬת��Ϊ����
    std::vector<float> tensor = imageToTensor(imagePath, imageHeight, imageWidth);

    std::cout << tensor.size() << std::endl;

    // --- 1. ����ģ�Ͳ��� ---
    const std::string engine_file = "E:\\work2\\pythonProjects\\CompressAI-master\\examples\\bmshj2018_factorized_encoder_2.engine";
    // const std::string engine_file = "E:/work2/c/engine_runtime/bmshj2018_factorized_encoder.trt";
    const char *input_name = "input";
    const char *z_hat_name = "z_hat";
    const char *likelihoods_name = "likelihoods";

    // --- 2. ��ȡ TensorRT Engine �ļ� ---
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "�޷��� Engine �ļ�: " << engine_file << std::endl;
        return 1;
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    if (buffer.empty())
    {
        std::cerr << "Engine �ļ�Ϊ��" << std::endl;
        return 1;
    }

    // --- 3. ���� TensorRT Runtime ---
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    CHECK_PTR(runtime, "���� TensorRT Runtime ʧ��");

    // --- 4. �����л� Engine ---
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    CHECK_PTR(engine, "�����л� Engine ʧ��");

    // --- 5. ����ִ�������� ---
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    CHECK_PTR(context, "����ִ��������ʧ��");

    // --- 6. ��ȡ������Ϣ ---
    int nbTensors = engine->getNbIOTensors();
    std::cout << "��������: " << nbTensors << std::endl;
    std::map<std::string, void *> tensor_buffers;
    std::map<std::string, int> tensor_sizes;
    std::map<std::string, int> tensor_indices;        // �������Ƶ���������ӳ��
    std::vector<void *> bindings(nbTensors, nullptr); // ������

    bool found_input = false, found_z_hat = false, found_likelihoods = false;
    for (int i = 0; i < nbTensors; ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        auto shape = engine->getTensorShape(tensor_name);
        int size = 1;
        std::cout << "���� " << i << ": " << tensor_name << " (";
        std::cout << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "����" : io_mode == nvinfer1::TensorIOMode::kOUTPUT ? "���"
                                                                                                                      : "����");
        std::cout << "), ��״: [";
        for (int j = 0; j < shape.nbDims; ++j)
        {
            size *= shape.d[j];
            std::cout << shape.d[j] << (j < shape.nbDims - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // ���仺����
        void *buffer = nullptr;
        CHECK_CUDA(cudaMalloc(&buffer, size * imageHeight * imageWidth * sizeof(float)));
        tensor_buffers[tensor_name] = buffer;
        tensor_sizes[tensor_name] = size * imageHeight * imageWidth;
        tensor_indices[tensor_name] = i; // ��¼��������
        bindings[i] = buffer;            // ���ð�����

        // ��֤Ԥ������
        if (std::strcmp(tensor_name, input_name) == 0)
        {
            found_input = true;
            if (io_mode != nvinfer1::TensorIOMode::kINPUT)
            {
                std::cerr << "����: " << tensor_name << " ������������" << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(tensor_name, z_hat_name) == 0)
        {
            found_z_hat = true;
            if (io_mode != nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cerr << "����: " << tensor_name << " �����������" << std::endl;
                return 1;
            }
        }
        else if (std::strcmp(tensor_name, likelihoods_name) == 0)
        {
            found_likelihoods = true;
            if (io_mode != nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cerr << "����: " << tensor_name << " �����������" << std::endl;
                return 1;
            }
        }
    }

    // ��֤����Ԥ������
    if (!found_input || !found_z_hat || !found_likelihoods)
    {
        std::cerr << "����: δ�ҵ�����Ԥ������: "
                  << "input=" << (found_input ? "�ҵ�" : "δ�ҵ�") << ", "
                  << "z_hat=" << (found_z_hat ? "�ҵ�" : "δ�ҵ�") << ", "
                  << "likelihoods=" << (found_likelihoods ? "�ҵ�" : "δ�ҵ�") << std::endl;
        std::cerr << "������ trtexec --verbose �����������" << std::endl;
        for (auto &buf : tensor_buffers)
        {
            CHECK_CUDA(cudaFree(buf.second));
        }
        return 1;
    }

    // --- 7. ׼���������� ---
    // int input_size = tensor_sizes[input_name];
    // std::vector<float> input_data(input_size, 1.0f); // ʾ������
    // CHECK_CUDA(cudaMemcpy(tensor_buffers[input_name], tensor.data(), 1 * 3 * imageHeight * imageWidth * sizeof(float), cudaMemcpyHostToDevice));

    // // --- 8. ����������ַ ---
    // for (const auto &buf : tensor_buffers)
    // {
    //     if (!context->setTensorAddress(buf.first.c_str(), buf.second))
    //     {
    //         std::cerr << "����������ַʧ��: " << buf.first << std::endl;
    //         for (auto &b : tensor_buffers)
    //         {
    //             CHECK_CUDA(cudaFree(b.second));
    //         }
    //         return 1;
    //     }
    // }

    // // --- 9. ִ������ ---
    // if (!context->executeV2(bindings.data()))
    // {
    //     std::cerr << "����ִ��ʧ��" << std::endl;
    //     for (auto &buf : tensor_buffers)
    //     {
    //         CHECK_CUDA(cudaFree(buf.second));
    //     }
    //     return 1;
    // }

    // // --- 10. ��ȡ��� ---
    // int z_hat_size = tensor_sizes[z_hat_name];
    // int likelihoods_size = tensor_sizes[likelihoods_name];
    // std::vector<float> z_hat_data(z_hat_size);

    // std::vector<float> likelihoods_data(likelihoods_size);
    // try
    // {
    //     CHECK_CUDA(cudaMemcpy(z_hat_data.data(), tensor_buffers[z_hat_name], z_hat_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA(cudaMemcpy(likelihoods_data.data(), tensor_buffers[likelihoods_name], likelihoods_size * sizeof(float), cudaMemcpyDeviceToHost));
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "�����������ʧ��: " << e.what() << std::endl;
    //     for (auto &buf : tensor_buffers)
    //     {
    //         CHECK_CUDA(cudaFree(buf.second));
    //     }
    //     return 1;
    // }

    // // --- 11. ��ӡ��� ---
    // std::cout << "������ɣ��������:" << std::endl;
    // std::cout << "z_hat[0]: " << z_hat_data[0] << std::endl;
    // std::cout << "likelihoods[0]: " << likelihoods_data[0] << std::endl;

    // // --- 12. ���� ---
    // for (auto &buf : tensor_buffers)
    // {
    //     CHECK_CUDA(cudaFree(buf.second));
    // }

    std::cout << "���������˳�" << std::endl;

    return 0;
}