#include "../includes/image.h"
#include "cmath"

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
                // if (w < left || h < top || w >= width - right || h >= height - bottom)
                //     tensor[c * height * width + h * width + w] = 0.00;
                // else
                tensor[c * height * width + h * width + w] = channels[c].at<float>(h, w);
            }
        }
    }

    return tensor;
}

std::vector<float> getImagePart(std::vector<cv::Mat> channels, int x0, int x1, int y0, int y1)
{
    // 分配张量 (1, 3, height, width)
    int width = (x1 - x0), height = (y1 - y0);
    std::vector<float> tensor(1 * 3 * width * height);

    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                // if (w < left || h < top || w >= width - right || h >= height - bottom)
                //     tensor[c * height * width + h * width + w] = 0.00;
                // else
                tensor[c * height * width + h * width + w] = channels[c].at<float>(y0 + h, x0 + w);
            }
        }
    }

    return tensor;
}

void getRawPart(std::vector<float> datas, std::vector<float> &tensor, int x0, int x1, int y0, int y1, int orig_h, int orig_w)
{

    // 分配张量 (1, 3, height, width)
    int width = (x1 - x0), height = (y1 - y0);

    std::vector<float> temp(1 * width * height);

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            temp[h * width + w] = datas[(y0 + h) * orig_w + (x0 + w)];
        }
    }

    tensor.clear();
    tensor.reserve(1 * 3 * width * height);

    for (size_t i = 0; i < 3; i++)
    {
        tensor.insert(tensor.end(), temp.begin(), temp.end());
    }
}

// 拆分图片
// image_path - 图片地址
// patch_size - 每块大小
void split_image(std::string image_path, std::vector<std::vector<float>> &datas, nlohmann::json &patchInfo, int &height, int &width, int patch_size)
{

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    height = img.rows;
    width = img.cols;
    // 转换为 RGB（OpenCV 默认 BGR）
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 归一化到 [0, 1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // 通道分离
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    int y = 0;

    // nlohmann::json j = nlohmann::json::array();

    while (y < height)
    {
        int endY = MIN(y + patch_size, height);
        int x = 0;
        while (x < width)
        {

            int endX = MIN(x + patch_size, width);
            std::vector<float> d = getImagePart(channels, x, endX, y, endY);
            datas.push_back(d);
            patchInfo.push_back({
                {"y", y},
                {"x", x},
                {"height", (endY - y)},
                {"width", (endX - x)},
            });
            x = x + patch_size;
        }
        y = y + patch_size;
    };
}
