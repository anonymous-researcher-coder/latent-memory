import json
import logging
from pathlib import Path
import torch
from safetensors import safe_open
from .memory_compressor import validate_compressor_checkpoint

PREFIXES = ("prompt_compressor.", "inference_compressor.")


def load_memgen_checkpoint(model, directory):
    directory = Path(directory).resolve()
    saved = json.loads((directory / "config.json").read_text())
    validate_compressor_checkpoint(saved, model.config)
    for key in (
        "prompt_latents_len",
        "inference_latents_len",
        "weaver_memory_layer",
        "weaver_random_memory_layers",
        "trigger_num_actions",
    ):
        if key in saved and saved[key] != getattr(model.config, key, None):
            raise ValueError(
                f"Checkpoint {key} mismatch: {saved[key]!r} != {getattr(model.config, key, None)!r}"
            )
    index_path = directory / "model.safetensors.index.json"
    index = (
        json.loads(index_path.read_text())["weight_map"]
        if index_path.exists()
        else None
    )
    shard_names = sorted(set(index.values())) if index else ["model.safetensors"]
    expected = model.state_dict(keep_vars=True)
    groups = {}
    for name, value in expected.items():
        groups.setdefault(id(value), []).append(name)
    aliases = {name: names for names in groups.values() for name in names}
    locations = {}
    for shard in shard_names:
        path = (directory / shard).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Shard outside checkpoint directory")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise ValueError(f"Duplicate key: {key}")
                if index and index.get(key) != shard:
                    raise ValueError(f"Invalid shard index: {key}")
                locations[key] = (path, tuple(handle.get_slice(key).get_shape()))
    if index and set(index) != set(locations):
        raise ValueError("Missing indexed tensors")
    allowed_new = (
        {k for k in expected if k.startswith(PREFIXES)}
        if not saved.get("compress_latent_memory", False)
        else set()
    )
    missing = [
        k
        for k in expected
        if not set(aliases[k]).intersection(locations) and k not in allowed_new
    ]
    unexpected = sorted(set(locations) - set(expected))
    mismatched = [
        k
        for k, (_, shape) in locations.items()
        if k in expected and tuple(expected[k].shape) != shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatched}"
        )
    copied = set()
    with torch.no_grad():
        for shard in shard_names:
            path = directory / shard
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    target = expected[key]
                    value = handle.get_tensor(key).to(
                        device=target.device, dtype=target.dtype
                    )
                    if id(target) in copied:
                        if not torch.equal(target, value):
                            raise ValueError(f"Conflicting shared tensor: {key}")
                    else:
                        target.copy_(value)
                        copied.add(id(target))
    logging.info(
        "Restored %d checkpoint tensors from %s; initialized %d compressor tensors",
        len(locations),
        directory,
        len(allowed_new),
    )
    return model
