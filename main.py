import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', torch.cuda.get_device_name(0) if device.type == 'cuda' else device.type)

if __name__ == '__main__':
    print(torch.rand(4, device=device))
