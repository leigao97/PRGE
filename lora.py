from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, based_layer: nn.Module, in_features: int, out_features: int, r: int, lora_alpha: int, lora_dropout: float, n: int, zo_mode):
        super(LoRALinear, self).__init__()

        self.base_layer = based_layer

        self.in_features = in_features
        self.out_features = out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.n = n

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        if zo_mode == "single":
            self.dup = n
        elif zo_mode == "dual":
            self.dup = 2*n
        else:
            raise ValueError(f"Unknown zo_mode: {zo_mode}")
    
        # Get the dtype of the base layer
        self.dtype = based_layer.weight.dtype
        
        self.lora_A = nn.Parameter(torch.zeros((r, in_features), dtype=self.dtype, device=self.base_layer.weight.device))
        self.lora_B = nn.Parameter(torch.zeros((self.dup, out_features, r), dtype=self.dtype, device=self.base_layer.weight.device))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        if self.training:
            frozen_out = self.base_layer(x)

            bsz = x.size(0)
            seq_len = x.size(1)

            lora_out = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
            lora_out = torch.matmul(lora_out.view(self.dup, bsz // (self.dup), seq_len, self.r), 
                                    self.lora_B.view(self.dup, 1, self.out_features, self.r).transpose(2, 3)) * self.scaling
            lora_out = lora_out.reshape(bsz, seq_len, self.out_features)
            return frozen_out + lora_out
        else:
            frozen_out = self.base_layer(x)
            lora_out = x @ self.lora_A.transpose(0, 1) @ self.lora_B[0].transpose(0, 1) * self.scaling
            return frozen_out + lora_out

@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    n: int
    target_modules: List[str]
    zo_mode: str
    dropout: float = 0.0

def get_peft_model(model, lora_config: LoraConfig):
    r = lora_config.r
    lora_alpha = lora_config.lora_alpha
    n = lora_config.n
    target_modules = lora_config.target_modules
    zo_mode = lora_config.zo_mode
    dropout = lora_config.dropout

    replace_with_lora(model, r, lora_alpha, dropout, n, target_modules, zo_mode)
    return model

def replace_with_lora(model, r, lora_alpha, lora_dropout, n, target_modules, zo_mode):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in target_modules:
            layer = LoRALinear(child, child.in_features, child.out_features,
                         r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                         n=n, zo_mode=zo_mode)
            setattr(model, name, layer)
        else:
            replace_with_lora(child, r, lora_alpha, lora_dropout, n, target_modules, zo_mode)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if "lora_B" in name:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )