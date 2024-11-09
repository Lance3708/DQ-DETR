import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda()