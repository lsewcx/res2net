import torch
from mmpretrain.models import Res2Net

model = Res2Net(depth=50, scales=4, base_width=26,out_indices=(0, 1, 2, 3))
model.eval()
inputs = torch.rand(1, 3, 256, 256)
outputs = model(inputs)
print(outputs)  # torch.Size([1, 1000])