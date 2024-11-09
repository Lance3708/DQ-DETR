import os
import torch

# 查看CUDA_VISIBLE_DEVICES变量
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# 检查可用的GPU数量
print("Number of available GPUs:", torch.cuda.device_count())

# 打印每个GPU设备的名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")