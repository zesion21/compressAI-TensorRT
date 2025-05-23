# compressAI-TensorRT

### 描述

A demo，loading .trt model from onnx

```
project
│   README.md
│   main.cpp //程序主文件
│
|—— src
|
|—— includes
│
|—— data
|       1.jpg //一张1024*1024的测试文件
|
|—— models
|       entropy_cdf.json // bmshj2018_factorized 模型的cdf
|       bmshj2018_factorized_encoder_self.onnx // bmshj2018_factorized生成的onnx
|       encoder_768.trt // 只支持 512*768
|       encoder1024.trt // 只支持 1024*1024
|       encoder_max_2048.trt // 64*64 ~ 2048*2048
|—— out //测试输出
|
|—— pyDemo //用py还原的代码
|       MyTest.py //一个简单的图片的压缩和还原
|       withSplit.py // 带大图拆分的大图压缩和还原（解决内存不足的问题）
|       splitRaw.py //一个拆分二进制存储图片的demo

```

### 环境说明

- CUDA 12.9
- TensorRT-10.10.0.31.Windows.win10.cuda-12.9 [下载 TensorRT ](https://developer.nvidia.com/tensorrt/download)
- CUDA Toolkit 12.9 [下载 CUDA Toolkit ](https://developer.nvidia.com/cuda-downloads)
- 使用的 compressAI 模型为： `bmshj2018_factorized` 质量为：`8` 编码器 `ans`

### 操作流程

#### 1. 导出 onnx 模型以及提取模型对应的 cdf

参考 ： [https://github.com/zesion21/compressAI-onnx/blob/main/README.md](https://github.com/zesion21/compressAI-onnx/blob/main/README.md)

#### 2. 使用 trxexec 将 onnx 模型转为 `.trt`模型

动态形状支持

```bash
trtexec --onnx=bmshj2018_factorized_encoder.onnx   --saveEngine=encoder.trt --fp16 \
  --minShapes=input:1x3x256x256 \
  --optShapes=input:1x3x512x512 \
  --maxShapes=input:16x3x2048x2048 \
```

- `--minShape`s：最小输入形状（例如，batch size=1）。
- `--optShapes`：优化输入形状（常用形状，例如 batch size=8）。
- `--maxShapes`：最大输入形状（例如，batch size=16）。
- `input` 是模型输入的名称（可通过 Netron 查看）。

注意：采用动态形状后，推理时必须指定输入的 shape :

```c++
context->setInputShape("input", nvinfer1::Dims4{1, 3, height, width});
```

测试转换完成的模型能否推理指定形状的输入：

```bash
trtexec \
  --loadEngine=encoder.trt \
  --shapes=input:1x3x256x256
```
