import torch

x = torch.tensor([[[1.25, -0.75, 2.5, -4.0], [0.002, 0.0, -0.003, 0.005]]], dtype=torch.float32)
gamma = torch.tensor([0.25, -1.5, 0.8, 2.25], dtype=torch.float32)
y = torch.nn.functional.rms_norm(x, (4,), gamma, 1e-5)
print(",".join(f"{v:.10f}" for v in y.flatten().tolist()))
