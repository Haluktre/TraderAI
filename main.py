import torch
import time

device = torch.device("mps")

x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

start = time.time()
z = torch.matmul(x, y)
end = time.time()

print(end-start)