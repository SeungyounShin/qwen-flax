from typing import Callable, List, Optional, Tuple, Union, Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn
from flax.linen.attention import dot_product_attention
import numpy as np

from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


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


def rotate_half(x):
    """Rotates half the hidden dims of the input.

    Args:
        x (jnp.ndarray): Input tensor.
    
    Returns:
        jnp.ndarray: Tensor with half dimensions rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (jnp.ndarray): The query tensor.
        k (jnp.ndarray): The key tensor.
        cos (jnp.ndarray): The cosine part of the rotary embedding.
        sin (jnp.ndarray): The sine part of the rotary embedding.
        position_ids (jnp.ndarray, optional): Deprecated and unused.
        unsqueeze_dim (int, optional): The dimension along which to expand cos and sin for broadcasting.
        
    Returns:
        tuple(jnp.ndarray, jnp.ndarray): The rotated query and key tensors.
    """
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlaxQwen2Attention(nn.Module):
    config: Qwen2Config
    layer_idx: int
    deterministic: bool = True

    def setup(self):
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Dense(features=self.config.num_attention_heads * self.head_dim, use_bias=True)
        self.k_proj = nn.Dense(features=self.config.num_key_value_heads * self.head_dim, use_bias=True)
        self.v_proj = nn.Dense(features=self.config.num_key_value_heads * self.head_dim, use_bias=True)
        self.o_proj = nn.Dense(features=self.config.num_attention_heads * self.head_dim, use_bias=False)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: Tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray],
        past_key_value: Optional[Any] = None,
        cache_position : Optional[jnp.ndarray] = None,
        **kwargs,      
    ):
        input_shape = hidden_states.shape[:-1]
        # --- 선형 투영 후 reshape ---
        # q_proj: [batch, seq_len, num_attention_heads * head_dim]
        # reshape하여 [batch, seq_len, num_attention_heads, head_dim]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        num_attention_heads = self.config.num_attention_heads
        q = q.reshape(input_shape + (num_attention_heads, self.head_dim))
        # k, v는 key/value head 개수가 다를 수 있음
        k = k.reshape(input_shape + (self.config.num_key_value_heads, self.head_dim))
        v = v.reshape(input_shape + (self.config.num_key_value_heads, self.head_dim))

        q = jnp.transpose(q, (0, 2, 1, 3)) # [batch, num_heads, seq_len, head_dim]
        k = jnp.transpose(k, (0, 2, 1, 3)) # [batch, num_heads, seq_len, head_dim]
        v = jnp.transpose(v, (0, 2, 1, 3)) # [batch, num_heads, seq_len, head_dim]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # attention 연산 수행
        print(attention_mask.shape)
        attn_output, attn_weights = dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attention_mask,
            dropout_rng=None,
            dropout_rate=self.attention_dropout,
            deterministic=self.deterministic,
            dtype=None,
            precision=None,
            # sliding_window=sliding_window, # Flax does not support sliding window
        )

        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(input_shape + (num_attention_heads * self.head_dim,))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



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

    # Attention 모듈 테스트
    
    # 저장된 PyTorch 데이터 로드
    q_proj_weight = np.load("./save_input_output_weight/q_proj_weight.npy")
    q_proj_bias = np.load("./save_input_output_weight/q_proj_bias.npy")
    k_proj_weight = np.load("./save_input_output_weight/k_proj_weight.npy")
    k_proj_bias = np.load("./save_input_output_weight/k_proj_bias.npy")
    v_proj_weight = np.load("./save_input_output_weight/v_proj_weight.npy")
    v_proj_bias = np.load("./save_input_output_weight/v_proj_bias.npy")
    out_proj_weight = np.load("./save_input_output_weight/out_proj_weight.npy")
    
    # 입력 데이터 로드
    hidden_states = np.load("./save_input_output_weight/hidden_states_input.npy")
    attention_mask = np.load("./save_input_output_weight/attention_mask_input.npy")
    position_embeddings_0 = np.load("./save_input_output_weight/position_embeddings_input_0.npy")
    position_embeddings_1 = np.load("./save_input_output_weight/position_embeddings_input_1.npy")
    expected_output = np.load("./save_input_output_weight/expected_output.npy")
    
    # JAX 배열로 변환
    q_proj_weight_jax = jnp.array(q_proj_weight)
    q_proj_bias_jax = jnp.array(q_proj_bias)
    k_proj_weight_jax = jnp.array(k_proj_weight)
    k_proj_bias_jax = jnp.array(k_proj_bias)
    v_proj_weight_jax = jnp.array(v_proj_weight)
    v_proj_bias_jax = jnp.array(v_proj_bias)
    out_proj_weight_jax = jnp.array(out_proj_weight)
    
    hidden_states_jax = jnp.array(hidden_states)
    attention_mask_jax = jnp.array(attention_mask)
    position_embeddings = (jnp.array(position_embeddings_0), jnp.array(position_embeddings_1))
    
    # 설정 파일 로드
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("save_input_output_weight/qwen_config")
    
    # Attention 모듈 초기화
    attention = FlaxQwen2Attention(config=config, layer_idx=0)
    
    # 파라미터 설정
    variables = {'params': {
        'q_proj': {'kernel': q_proj_weight_jax.T, 'bias': q_proj_bias_jax},
        'k_proj': {'kernel': k_proj_weight_jax.T, 'bias': k_proj_bias_jax},
        'v_proj': {'kernel': v_proj_weight_jax.T, 'bias': v_proj_bias_jax},
        'o_proj': {'kernel': out_proj_weight_jax.T}
    }}
    
    # 출력 계산
    output, _ = attention.apply(
        variables, 
        hidden_states_jax, 
        position_embeddings, 
        attention_mask_jax,
        deterministic=True
    )
    
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

    # MLP 테스트 코드는 그대로 유지
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
