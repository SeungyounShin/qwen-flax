from flax import nnx

from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring


class FlaxQwen2PreTrainedModel(FlaxPreTrainedModel):
    pass

class FlaxQwen2ForCausalLM(FlaxQwen2PreTrainedModel):
    pass


if __name__ == "__main__":
    
    model = FlaxQwen2ForCausalLM.from_pretrained("Qwen/Qwen2-72B-Instruct")
