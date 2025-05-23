import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_factorized
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import os
import time

from readRaw import doReadRaw

from splitRaw import splitRaw2Np

Image.MAX_IMAGE_PIXELS = None

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = bmshj2018_factorized(quality=8, pretrained=True).to(device).eval()


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding."""
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


def split_image(img_np, patch_size=512):
    """将图像分割为多个小块，返回小块列表和分割信息"""
    height, width, _ = img_np.shape
    patches = []
    patch_info = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            patch = img_np[y:y_end, x:x_end, :]

            # 填充到 patch_size
            patch_h, patch_w = patch.shape[:2]
            # if patch_h < patch_size or patch_w < patch_size:
            #     padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            #     padded_patch[:patch_h, :patch_w, :] = patch
            #     patch = padded_patch
            # else:
            #     padded_patch = patch

            patches.append(patch)
            patch_info.append(
                {
                    "y": y,
                    "x": x,
                    "height": patch_h,
                    "width": patch_w,
                    # "padded_height": patch.shape[0],
                    # "padded_width": patch.shape[1],
                }
            )

    return patches, patch_info, height, width


def doPressRaw(rawPath: str, outPath: str, patch_size=2048, size=10240):
    """size—— 每块raw的高度大小"""
    column = 11936
    state = os.stat(rawPath)
    count = state.st_size // (size * 23872)
    for i in range(count):
        print(f"开始处理第{i+1}个")
        out = splitRaw2Np(rawPath, index=i, size=size, offset=i * column * size * 2)
        doPressAndSave(out, f"{outPath}_{i}", patch_size)

    last_byte = state.st_size % (size * 23872)
    if last_byte != 0:
        last_size = last_byte // 23872
        print(f"开始处理剩余部分")
        out = splitRaw2Np(
            rawPath, count, size=last_size, offset=count * column * size * 2
        )
        doPressAndSave(out, f"{outPath}_{count}", patch_size)


def doPressAndSave(img_np, outPath, patch_size=2048):
    start = time.time()
    orig_h, orig_w = img_np.shape[:2]

    # 分割图像
    patches, patch_info, _, _ = split_image(img_np, patch_size)

    # 压缩每个小块
    all_strings = []
    all_shapes = []
    all_string_lengths = []
    for patch in patches:
        # 转换为张量
        x = transforms.ToTensor()(patch).unsqueeze(0).to(device)
        h, w = x.size(2), x.size(3)
        pad, _ = compute_padding(h, w, min_div=64)
        x = F.pad(x, pad, mode="constant", value=0)

        # 压缩
        with torch.no_grad():
            out = net.compress(x)
        strings = out["strings"]
        shape = out["shape"]

        # 存储二进制流及其长度
        patch_strings = [s[0] for s in strings]  # 假设单一字符串，视模型调整
        all_strings.append(patch_strings)
        all_shapes.append(shape)
        all_string_lengths.append([len(s) for s in patch_strings])

    # 保存二进制数据
    bin_path = f"{outPath}.bin"
    with open(bin_path, "wb") as f:
        for strings in all_strings:
            for s in strings:
                f.write(s)

    # 保存 JSON 信息
    json_path = f"{outPath}.json"
    json_data = {
        "original_height": orig_h,
        "original_width": orig_w,
        "patch_size": patch_size,
        "patches": [
            {
                "shape": shape,
                "y": info["y"],
                "x": info["x"],
                "height": info["height"],
                "width": info["width"],
                "string_lengths": lengths,
            }
            for shape, info, lengths in zip(all_shapes, patch_info, all_string_lengths)
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"压缩完成！耗时：{(time.time()-start):.4f}秒，输出：{bin_path}, {json_path}")


def doPress(inputPath: str, outPath: str, patch_size=2048, size=10240):
    print("开始进行压缩操作！")
    start = time.time()
    _, file_extension = os.path.splitext(inputPath)
    if ".raw" == file_extension:
        doPressRaw(inputPath, outPath, patch_size, size)
    else:
        # 走图片压缩
        img = Image.open(inputPath).convert("RGB")
        img_np = np.array(img).astype(np.uint8)
        doPressAndSave(img_np, outPath, patch_size)
    print(f"文件压缩完成耗时：{(time.time()-start):.4f}秒")


def doDepress(input: str, out: str):
    # 读取 JSON 信息
    start = time.time()
    json_path = f"{os.path.splitext(input)[0]}.json"
    with open(json_path, "r") as f:
        json_data = json.load(f)

    orig_h = json_data["original_height"]
    orig_w = json_data["original_width"]
    patches_info = json_data["patches"]

    # 初始化输出图像
    recon_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    # 读取二进制文件
    with open(input, "rb") as f:
        bin_data = f.read()

    offset = 0
    for patch_info in patches_info:
        shape = torch.tensor(patch_info["shape"]).to(device)
        y, x = patch_info["y"], patch_info["x"]
        height, width = patch_info["height"], patch_info["width"]
        string_lengths = patch_info["string_lengths"]

        # 读取二进制流
        strings = []
        for length in string_lengths:
            strings.append([bin_data[offset : offset + length]])
            offset += length

        # 解压缩
        with torch.no_grad():
            x_hat = net.decompress(strings, shape)["x_hat"]

        H, W = x_hat.size(2), x_hat.size(3)
        _, unpad = compute_padding(height, width, out_h=H, out_w=W)
        x_hat = F.pad(x_hat, unpad, mode="constant", value=0)

        # 转换为 NumPy 数组
        patch_img = transforms.ToPILImage()(x_hat.clamp_(0, 1).squeeze())
        patch_np = np.array(patch_img)[:height, :width, :]
        recon_img[y : y + height, x : x + width, :] = patch_np

    # 保存输出图像
    Image.fromarray(recon_img).save(out)
    end = time.time()
    print(f"解压缩完成,耗时：{(end-start):.4f}秒！输出：{out}")


if __name__ == "__main__":
    #     # 示例用法
    #     # name = "stmalo_fracape"
    #     input_png = f"examples/assets/stmalo_fracape.png"
    #     output_prefix = "examples/assets/compressed"
    #     output_recon_png = "examples/assets/reconstructed2.png"
    patch_size = 2048  # 调整以适应 GPU 内存

    #     start = time.time()
    #     print("开始处理")
    #     # 压缩
    # doPress(
    #     "E:/work2/c/engine_runtime/out/shiyan.png",
    #     "E:/work2/c/engine_runtime/out/compressed",
    #     patch_size=patch_size,
    #     size=9433,
    # )

#     # 解压缩
doDepress(
    "E:/work2/c/engine_runtime/out/compress.bin",
    "E:/work2/c/engine_runtime/out/compress.png",
)

#     print(f"处理完成，耗时：{(time.time()-start):.4f}")
