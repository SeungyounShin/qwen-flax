import torch
from transformers import AutoModelForCausalLM
import numpy as np
import os


qwen_pt = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    attn_implementation="eager",
)

input_ids = torch.tensor([[1, 2, 3, 4, 5]])

output = qwen_pt(input_ids)

print(output)   

