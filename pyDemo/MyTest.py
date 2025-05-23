from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from compressai.zoo import bmshj2018_factorized
import pickle
import os

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
net = bmshj2018_factorized(quality=8, pretrained=True).to(device).eval()
Image.MAX_IMAGE_PIXELS = None


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
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


def doPress(inputPath: str, outPath: str):
    # 图片转张量
    img = Image.open(inputPath).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    h, w = x.size(2), x.size(3)
    pad, _ = compute_padding(h, w, min_div=64)
    x = F.pad(x, pad, mode="constant", value=0)
    # 初始化模型

    with torch.no_grad():
        out = net.compress(x)
    with open(f"{outPath}_{w}_{h}.bin", "wb") as f:
        pickle.dump({"shape": out["shape"], "strings": out["strings"]}, file=f)
    print("压缩完成！")


def doDepress(input: str, out: str):

    basename = os.path.basename(input)
    name = os.path.splitext(basename)[0]
    ss = name.split("_")
    w = int(ss[-2])
    h = int(ss[-1])
    with open(input, "rb") as file:
        # data = pickle.load(file=file)
        data = file.read()
    with torch.no_grad():

        # x_hat = net.decompress(data["strings"], data["shape"])["x_hat"]
        x_hat = net.decompress([[data]], [64, 84])["x_hat"]

    # H, W = x_hat.size(2), x_hat.size(3)
    # _, unpad = compute_padding(h, w, out_h=H, out_w=W)
    # x_hat = F.pad(x_hat, unpad, mode="constant", value=0)

    img = transforms.ToPILImage()(x_hat.clamp_(0, 1).squeeze())
    img.save(out)
    print("解压缩完成")
    return "解压缩完成"


import time

if __name__ == "__main__":
    start = time.time()

    # name = "stmalo_fracape"
    # inputPath = f"examples/assets/{name}.png"
    # outPath = f"examples/assets/{name}_bin"
    # doPress(inputPath, outPath)

    input = f"examples/assets/onnx_stmalo_fracape_768_512.bin"
    output = f"examples/assets/1323_995.png"
    doDepress(
        "E:/work2/c/engine_runtime/out/1_1344_1024.bin",
        "E:/work2/c/engine_runtime/out/1_1024_1024.png",
    )
    # doDepress(input, "examples/assets/1.png")

    end = time.time()
    print(f"耗时：{(end-start):.4f}")
