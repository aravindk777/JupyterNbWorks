import torch

print(f"Torch version: {torch.version.__version__}")
print(f"Is CUDA built: {torch.backends.cuda.is_built()}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {device}")
