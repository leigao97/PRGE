import torch
from torch import nn
import math


class LoRALinear(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 4, r: int = 2, lora_alpha: int = 2):
        # nn.Linear.__init__(self, in_features, out_features, **kwargs)
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.register_buffer("lora_A", torch.zeros((r, in_features)))
        self.register_buffer("lora_B", torch.zeros((2, 1, r, out_features)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, learning_rate: float = 0.1, eps: float = 1e-8, projected_grad: float = 0.1):
        lora_B0 = self.lora_B[0].data
        lora_B1 = self.lora_B[1].data

        noise = (lora_B0 - lora_B1) / 2
        z1 = (learning_rate * projected_grad) * noise / eps
        z2 = eps * torch.randn(size=lora_B0.size())

        # Combining the changes
        lora_B0_changes = z2 - noise - z1
        lora_B1_changes = noise - z1 - z2

        # Applying the changes
        lora_B0.add_(lora_B0_changes)
        lora_B1.add_(lora_B1_changes)
        
        intermediate = x @ self.lora_A.T

        bsz = x.size(0)
        seq_len = x.size(1)
        embedding_dim = x.size(2)

        lora_out = torch.matmul(intermediate.reshape(2, bsz // 2, seq_len, self.r), self.lora_B) * self.scaling

        return lora_out.reshape(bsz, seq_len, embedding_dim)