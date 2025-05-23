import numpy as np
import os
from PIL import Image
import time

baseUrl = "F:/compressAI/test_data/"
column = 11936
xReadSize = 9520


def splitRaw(path: str, size=10240, index=0):
    # with open(path, "rb") as f:
    #     buffer = f.read()
    # print(len(buffer))
    raw = np.fromfile(
        path, dtype=np.uint8, count=23872 * size, offset=index * 23872 * size
    )
    raw.tofile(f"{baseUrl}output_file_{index}.raw")


def splitRaw2Np(path: str, index=0, size: int = 9520, offset: int = 0):
    start  = time.time()
    tempPANdata = np.zeros((size, xReadSize), dtype=np.uint16)
    raw_data = np.fromfile(path, dtype=">u2", count=column * size, offset=offset)
    # 重塑为二维数组并截取有效区域
    raw_data = raw_data.reshape(-1, column)[:size, 32 : 32 + xReadSize]
    tempPANdata[:size, :xReadSize] = raw_data
    max_val = np.max(tempPANdata)
    min_val = np.min(tempPANdata)

    tempPANdata = (tempPANdata - min_val) / (max_val - min_val) * 255
    tempPANdata = tempPANdata.astype(np.uint8)
    out = np.stack([tempPANdata] * 3, axis=2)
    print(f"读取完第{index+1}个完成，耗时：{(time.time()-start):.4f}")
    return out
    # image = Image.fromarray(tempPANdata, mode="L")
    # image.save(f"{baseUrl}{index}.png", "PNG")


if __name__ == "__main__":
    rawPath = f"{baseUrl}decrypt1.raw"
    size = 9520
    state = os.stat(rawPath)
    count = state.st_size // (size * 23872)
    for i in range(count):
        print(f"开始处理第{i+1}个")
        splitRaw2Np(rawPath, index=i, size=size, offset=i * column * size * 2)

    last_byte = state.st_size % (size * 23872)
    if last_byte != 0:
        last_size = last_byte // 23872
        print(f"开始处理剩余部分")
        splitRaw2Np(rawPath, count, size=last_size, offset=count * column * size * 2)
