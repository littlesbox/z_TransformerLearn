import torch 
import torchvision 
print(f"PyTorch version: {torch.__version__}") 
print(f"TorchVision version: {torchvision.__version__}") 
print(f"Is MPS (Metal Performance Shaders) available? {torch.backends.mps.is_available()}") 
if torch.backends.mps.is_available(): 
    mps_device = torch.device("mps") 
    print(f"MPS device: {mps_device}") 
    # 创建一个张量并将其移动到MPS设备 
    x = torch.ones(2, 3, device=mps_device) 
    print(x) 
else: 
    print("MPS is not available. Please check your PyTorch installation.")