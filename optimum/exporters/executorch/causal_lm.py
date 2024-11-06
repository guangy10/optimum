from transformers import AutoModelForCausalLM, GenerationConfig
from .task_registry import register_task


@register_task("text-generation")
def load_causal_lm_model(model_name_or_path: str, **kwargs):
    device = "cpu"
    dtype = torch.float32
    cache_implementation = "static"
    attn_implementation = "sdpa"
    batch_size = 1
    max_generation_length = kwargs.get("max_length", 256)

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_generation_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_generation_length,
            },
        ),
    )
