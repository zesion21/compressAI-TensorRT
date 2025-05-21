#include "../includes/image.h"

void greet(std::string name)
{

    std::cout << "Hello " << name << std::endl;
}

std::vector<float> imageToTensor(const std::string &image_path, int &height, int &width)
{
    // 加载图片
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // 获取尺寸
    height = img.rows;
    width = img.cols;

    // 转换为 RGB（OpenCV 默认 BGR）
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 归一化到 [0, 1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // 分配张量 (1, 3, height, width)
    std::vector<float> tensor(1 * 3 * height * width);

    // 通道分离
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // 填充张量：R, G, B 顺序
    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                // float rounded = std::round(channels[c].at<float>(h, w) * 10000.0f) / 10000.0f;
                // tensor[c * height * width + h * width + w] = rounded;
                tensor[c * height * width + h * width + w] = channels[c].at<float>(h, w);
            }
        }
    }

    return tensor;
}

std::vector<float> imageToTensor1024(const std::string &image_path, int &height, int &width)
{
    // 加载图片
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    height = 1024;
    width = 1024;
    // 获取尺寸
    int image_height = img.rows;
    int image_width = img.cols;

    // 转换为 RGB（OpenCV 默认 BGR）
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 归一化到 [0, 1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // 分配张量 (1, 3, height, width)
    std::vector<float> tensor(1 * 3 * height * width);

    // 通道分离
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // 填充张量：R, G, B 顺序
    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                // float rounded = std::round(channels[c].at<float>(h, w) * 10000.0f) / 10000.0f;
                // tensor[c * height * width + h * width + w] = rounded;
                if (w < image_width && h < image_height)
                    tensor[c * height * width + h * width + w] = channels[c].at<float>(h, w);
                else
                    tensor[c * height * width + h * width + w] = 0.00;
            }
        }
    }

    return tensor;
}