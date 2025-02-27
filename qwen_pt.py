import torch
from transformers import AutoModelForCausalLM
import numpy as np

qwen_pt = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
qwen_pt.eval()

test_module = qwen_pt.model.layers[-1].mlp

gate_proj_weight = test_module.gate_proj.weight # torch.Size([4864, 896])
up_proj_weight = test_module.up_proj.weight # torch.Size([4864, 896])
down_proj_weight = test_module.down_proj.weight # torch.Size([896, 4864])

norm_test_input = torch.randn(4,2048,896)
expected_output = test_module(norm_test_input) # torch.Size([4, 2048, 896])

# save as numpy
## config
# qwen_pt.model.layers[-1].mlp.config.save_pretrained("save_input_output_weight/qwen_mlp_last_config")
# np.save("./save_input_output_weight/gate_proj_weight.npy", gate_proj_weight.detach().cpu().numpy())
# np.save("./save_input_output_weight/up_proj_weight.npy", up_proj_weight.detach().cpu().numpy())
# np.save("./save_input_output_weight/down_proj_weight.npy", down_proj_weight.detach().cpu().numpy())
# # save input and output
# np.save("./save_input_output_weight/norm_test_input.npy", norm_test_input.detach().cpu().numpy())
# np.save("./save_input_output_weight/norm_expected_output.npy", expected_output.detach().cpu().numpy())

import pdb; pdb.set_trace()
# load input
test_input = np.load("./save_input_output_weight/norm_test_input.npy")
test_input = torch.from_numpy(test_input)

# test gate_proj
print(test_module.gate_proj(test_input))