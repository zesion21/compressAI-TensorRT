#include "iostream"
#include "opencv2/opencv.hpp"
#include <vector>
#include <stdexcept>
#include "json.hpp"

void greet(std::string name);

std::vector<float> imageToTensor(const std::string &image_path, int &height, int &width);

std::vector<float> imageToTensor1024(const std::string &image_path, int &height, int &width);

void split_image(std::string image_path, std::vector<std::vector<float>> &datas, nlohmann::json &patchInfo, int &height, int &width, int patch_size);