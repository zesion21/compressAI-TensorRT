#include "iostream"
#include "opencv2/opencv.hpp"
#include <vector>
#include <stdexcept>
#include "json.hpp"

void greet(std::string name);

std::vector<float> imageToTensor(const std::string &image_path, int &height, int &width);

void split_image(std::string image_path, std::vector<std::vector<float>> &datas, nlohmann::json &patchInfo, int &height, int &width, int patch_size);

void getRawPart(std::vector<float> datas, std::vector<float> &tensor, int x, int endX, int y, int endY, int orig_h, int orig_w);