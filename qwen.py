import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn
import numpy as np

from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring

class FlaxQwen2MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    hidden_act: str

    def setup(self):
        self.gate_proj = nn.Dense(features=self.intermediate_size, use_bias=False)
        self.up_proj = nn.Dense(features=self.intermediate_size, use_bias=False)
        self.down_proj = nn.Dense(features=self.hidden_size, use_bias=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class FlaxQwen2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param("weight", lambda rng, shape: jnp.ones(shape, jnp.float32), (self.hidden_size,))

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        hidden_states = hidden_states.astype(input_dtype)
        return self.weight * hidden_states

    def __repr__(self):
        return f"FlaxQwen2RMSNorm(hidden_size={self.hidden_size}, eps={self.eps})"


class FlaxQwen2PreTrainedModel(FlaxPreTrainedModel):
    pass

class FlaxQwen2ForCausalLM(FlaxQwen2PreTrainedModel):
    pass


if __name__ == "__main__":
    
    # model = FlaxQwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")


    # debug Qwen2MLP
    
    # 저장된 PyTorch 데이터 로드
    gate_proj_weight = np.load("./save_input_output_weight/gate_proj_weight.npy")
    up_proj_weight = np.load("./save_input_output_weight/up_proj_weight.npy")
    down_proj_weight = np.load("./save_input_output_weight/down_proj_weight.npy")
    mlp_test_input = np.load("./save_input_output_weight/norm_test_input.npy")
    expected_output = np.load("./save_input_output_weight/norm_expected_output.npy")

    # JAX 배열로 변환
    gate_proj_weight_jax = jnp.array(gate_proj_weight)
    up_proj_weight_jax = jnp.array(up_proj_weight)
    down_proj_weight_jax = jnp.array(down_proj_weight)
    mlp_test_input_jax = jnp.array(mlp_test_input)

    # 크기 확인
    hidden_size = mlp_test_input.shape[-1]  # 896
    intermediate_size = gate_proj_weight.shape[0]  # 4864

    # MLP 모듈 초기화
    mlp = FlaxQwen2MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act="silu")

    # 파라미터 설정
    variables = {'params': {
        'gate_proj': {'kernel': gate_proj_weight_jax.T},
        'up_proj': {'kernel': up_proj_weight_jax.T},
        'down_proj': {'kernel': down_proj_weight_jax.T}
    }}
    import pdb; pdb.set_trace()
    # 출력 계산
    output = mlp.apply(variables, mlp_test_input_jax)

    # PyTorch와 JAX 출력 비교
    pt_output = expected_output
    jax_output = np.array(output)
    diff = np.abs(pt_output - jax_output).mean()

    print(f"PyTorch 출력 형태: {pt_output.shape}")
    print(f"JAX 출력 형태: {jax_output.shape}")
    print(f"평균 절대 차이: {diff}")
    print(f"최대 절대 차이: {np.abs(pt_output - jax_output).max()}")

    if diff < 1e-5:
        print("테스트 통과: JAX 구현이 PyTorch와 거의 동일합니다!")
    else:
        print("테스트 실패: JAX 구현이 PyTorch와 다릅니다.")
    
    # 일부 요소 출력하여 육안으로 비교
    print("\n샘플 값 비교:")
    sample_idx = (0, 0, 0)  # 첫 번째 요소
    print(f"PyTorch: {pt_output[sample_idx]}")
    print(f"JAX: {jax_output[sample_idx]}")
