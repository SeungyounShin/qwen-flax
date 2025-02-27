import torch
from transformers import AutoModelForCausalLM
import numpy as np
import os

qwen_pt = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
qwen_pt.eval()

test_module = qwen_pt.model.layers[0].self_attn

q_proj_w = test_module.q_proj.weight # torch.Size([896, 896])
q_proj_bias = test_module.q_proj.bias # torch.Size([896])
k_proj_w = test_module.k_proj.weight # torch.Size([896, 896])
k_proj_bias = test_module.k_proj.bias # torch.Size([896])
v_proj_w = test_module.v_proj.weight # torch.Size([896, 896])
v_proj_bias = test_module.v_proj.bias # torch.Size([896])
out_proj_w = test_module.o_proj.weight # torch.Size([896, 896])
config = test_module.config

hidden_states = torch.randn(1, 5, 896) #torch.Size([1, 5, 896])
attention_mask = torch.ones(1, 1, 5, 6)
position_ids = torch.arange(5)
position_embeddings = (torch.randn(1, 5, 64), torch.randn(1, 5, 64))

# 4) Self-Attention 모듈 테스트
with torch.no_grad():
    expected_output, _ = test_module(hidden_states = hidden_states, attention_mask = attention_mask, position_ids = position_ids, position_embeddings =  position_embeddings)

print("expected_output.shape:", expected_output.shape)
# 정상이라면 [4, 2048, 896].

# 가중치 및 입출력 저장
os.makedirs("save_input_output_weight", exist_ok=True)
# 설정 저장
qwen_pt.config.save_pretrained("save_input_output_weight/qwen_config")
# 가중치 저장
np.save("./save_input_output_weight/q_proj_weight.npy", q_proj_w.detach().cpu().numpy())
np.save("./save_input_output_weight/q_proj_bias.npy", q_proj_bias.detach().cpu().numpy())
np.save("./save_input_output_weight/k_proj_weight.npy", k_proj_w.detach().cpu().numpy())
np.save("./save_input_output_weight/k_proj_bias.npy", k_proj_bias.detach().cpu().numpy())
np.save("./save_input_output_weight/v_proj_weight.npy", v_proj_w.detach().cpu().numpy())
np.save("./save_input_output_weight/v_proj_bias.npy", v_proj_bias.detach().cpu().numpy())
np.save("./save_input_output_weight/out_proj_weight.npy", out_proj_w.detach().cpu().numpy())
# 입력 및 출력 저장
np.save("./save_input_output_weight/hidden_states_input.npy", hidden_states.detach().cpu().numpy())
np.save("./save_input_output_weight/attention_mask_input.npy", attention_mask.detach().cpu().numpy())
np.save("./save_input_output_weight/position_ids_input.npy", position_ids.detach().cpu().numpy())
np.save("./save_input_output_weight/position_embeddings_input_0.npy", position_embeddings[0].detach().cpu().numpy())
np.save("./save_input_output_weight/position_embeddings_input_1.npy", position_embeddings[1].detach().cpu().numpy())
np.save("./save_input_output_weight/expected_output.npy", expected_output.detach().cpu().numpy())

# 저장된 입력 로드 테스트
test_hidden_states = torch.from_numpy(np.load("./save_input_output_weight/hidden_states_input.npy"))
test_attention_mask = torch.from_numpy(np.load("./save_input_output_weight/attention_mask_input.npy"))
test_position_ids = torch.from_numpy(np.load("./save_input_output_weight/position_ids_input.npy"))
test_position_embeddings = (
    torch.from_numpy(np.load("./save_input_output_weight/position_embeddings_input_0.npy")),
    torch.from_numpy(np.load("./save_input_output_weight/position_embeddings_input_1.npy"))
)

# 로드된 입력으로 테스트
with torch.no_grad():
    test_output, _ = test_module(
        hidden_states=test_hidden_states, 
        attention_mask=test_attention_mask, 
        position_ids=test_position_ids, 
        position_embeddings=test_position_embeddings
    )

print("테스트 출력 형태:", test_output.shape)
print("원본 출력과 일치:", torch.allclose(expected_output, test_output))