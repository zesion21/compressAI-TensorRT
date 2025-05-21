#include "iostream"
#include "opencv2/opencv.hpp"
#include <vector>
#include <stdexcept>

void greet(std::string name);

std::vector<float> imageToTensor(const std::string &image_path, int &height, int &width);

std::vector<float> imageToTensor1024(const std::string &image_path, int &height, int &width);