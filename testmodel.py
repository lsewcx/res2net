import torch
from mmpretrain.models import Res2Net

# 创建模型并移动到CUDA设备上
model = Res2Net(depth=50, scales=4, base_width=26, out_indices=(0, 1, 2, 3)).cuda()
model.eval()

# 创建输入数据并移动到CUDA设备上
inputs = torch.rand(1, 3, 224, 224).cuda()

# 执行前向传播
outputs = model(inputs)

# 输出每个张量的形状
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}")