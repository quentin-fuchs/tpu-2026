"""Load Gemma-3 base model + LoRA adapter + tokenizer.

Two models are involved in GRPO:

  * reference  — the frozen base model. Used to compute KL(policy || reference)
                 so the policy doesn't drift away from sensible language.
  * actor      — the same base model wrapped with LoRA adapters. Only the
                 adapter weights are trainable. Smaller, cheaper, and keeps
                 KL naturally bounded.
"""
import json
import os

import jax
import qwix
from flax import nnx
from huggingface_hub import snapshot_download
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.checkpoint_manager import CheckpointManager
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

from config import (
    ALPHA,
    GEMMA_TOKENIZER_PATH,
    MESH,
    MODEL_ID,
    RANK,
)


def download_weights():
    """Snapshot the HF repo locally and read the EOS token IDs."""
    print(f"Downloading {MODEL_ID} from Hugging Face...")
    local_path = snapshot_download(repo_id=MODEL_ID, ignore_patterns=["*.pth"])
    print(f"Model downloaded to: {local_path}")

    eos_tokens = []
    cfg_path = os.path.join(local_path, "generation_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            eos_tokens = json.load(f).get("eos_token_id", [])
    return local_path, eos_tokens


def model_config_for(model_id: str):
    if "gemma-3-270m" in model_id:
        return gemma_lib.ModelConfig.gemma3_270m()
    if "gemma-3-1b" in model_id:
        return gemma_lib.ModelConfig.gemma3_1b_it()
    raise ValueError(f"Unknown model id: {model_id}")


def build_mesh():
    return jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))


def load_base_model(local_path: str, mesh):
    """Load the frozen base model (this is also the GRPO reference model)."""
    cfg = model_config_for(MODEL_ID)
    with jax.set_mesh(mesh):
        model = params_safetensors_lib.create_model_from_safe_tensors(local_path, cfg, mesh)
    return model, cfg


def get_lora_model(base_model, mesh):
    """Wrap the base model with LoRA adapters on attention + MLP projections."""
    lora_provider = qwix.LoraProvider(
        module_path=(
            r".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
            r".*attn_vec_einsum"
        ),
        rank=RANK,
        alpha=ALPHA,
    )
    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)

    # Re-shard after wrapping so the adapter weights live on the right devices.
    with jax.set_mesh(mesh):
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded)
    return lora_model


def load_lora_checkpoint(lora_model, ckpt_dir: str, step: int | None = None):
    """Restore LoRA adapter weights from a tunix checkpoint.

    ckpt_dir should be the directory that contains numbered step subdirectories,
    e.g. /home/.../results/ckpts/actor
    """
    mgr = CheckpointManager(root_directory=ckpt_dir)
    n, _ = mgr.maybe_restore(model=lora_model, step=step, restore_only_lora_params=True)
    if n == 0:
        raise RuntimeError(f"No checkpoint found under {ckpt_dir}.")
    print(f"Loaded LoRA checkpoint step={n} from {ckpt_dir}")
    return lora_model, n


def load_tokenizer(eos_tokens: list[int]):
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
    return tokenizer, eos_tokens
