# 指定CMake的最低版本
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(engine_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(CUDA_INCLUDE_DIRS "${CUDA_ROOT}/include")
set(CUDA_LIBRARY_DIRS "${CUDA_ROOT}/lib/x64")

set(TENSORRT_ROOT "D:/Program Files/TensorRT-10.10.0.31.Windows.win10.cuda-12.9")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARY_DIRS "${TENSORRT_ROOT}/lib")

set(OPENCV_ROOT "D:/Program Files/opencv/build")
set(OPENCV_ROOT_INCLUDE_DIRS "${OPENCV_ROOT}/include")

# set(TENSORRT_LIBRARY_DIRS "${TENSORRT_ROOT}/lib")
include_directories(${TENSORRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENCV_ROOT_INCLUDE_DIRS})

# 查找库
find_library(NVINFER_LIBRARY NAMES nvinfer_10 HINTS ${TENSORRT_LIBRARY_DIRS})
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin_10 HINTS ${TENSORRT_LIBRARY_DIRS})
find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser_10 HINTS ${TENSORRT_LIBRARY_DIRS})
find_library(CUDA_LIBRARY NAMES cudart HINTS ${CUDA_LIBRARY_DIRS})
find_library(OPENCV_LIBRARY NAMES opencv_world470 HINTS "D:/Program Files/opencv/build/x64/vc16/lib")

# 检查库
if(NOT NVINFER_LIBRARY OR NOT NVINFER_PLUGIN_LIBRARY OR NOT NVONNXPARSER_LIBRARY OR NOT CUDA_LIBRARY OR NOT OPENCV_LIBRARY)
    message(FATAL_ERROR "Required libraries not found. Check TENSORRT_ROOT and CUDA_ROOT.")
endif()

aux_source_directory(./includes DIR_INCLUDE)
aux_source_directory(./src DIR_SRCS)

# 添加可执行文件
add_executable(engine_runtime main.cpp ${DIR_SRCS} ${DIR_INCLUDE})

target_include_directories(engine_runtime PUBLIC
    ${CMAKE_SOURCE_DIR}/includes
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS} ${OPENCV_ROOT_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(engine_runtime
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${CUDA_LIBRARY}
    ${OPENCV_LIBRARY}
)
file(COPY ${CMAKE_SOURCE_DIR}/models/ DESTINATION ${CMAKE_BINARY_DIR}/models/)

# 复制 TensorRT DLL
set(TENSORRT_DLLS
    "${TENSORRT_ROOT}/bin/nvinfer_10.dll"
    "${TENSORRT_ROOT}/bin/nvinfer_plugin_10.dll"
    "${TENSORRT_ROOT}/bin/nvonnxparser_10.dll"
    "${OPENCV_ROOT}/x64/vc16/bin/opencv_world470.dll"
)

foreach(DLL ${TENSORRT_DLLS})
    add_custom_command(TARGET engine_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL}"
        $<TARGET_FILE_DIR:engine_runtime>)
endforeach()