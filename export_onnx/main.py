import torch
import torchvision

from denoising_diffusion.unet import *

dummy_input = torch.randn(1, 64, 64, 3, device="cuda")
t = torch.randn(1, 256, device="cuda")
residual_block = ResidualBlock(64, 64, 256).cuda()

torch.onnx.export(residual_block, 
                  (dummy_input,t), 
                  "./output/residual_block.onnx", 
                   verbose=True, 
                   input_names=["input_names"], 
                   output_names=["output_names"])