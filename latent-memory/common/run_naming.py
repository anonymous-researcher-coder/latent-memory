from datetime import datetime, timezone
import json
from pathlib import Path
import re


def _weaver_layer_count(model):
    name = model.get("weaver", {}).get("model_name") or model["model_name"]
    path = Path(name) / "config.json"
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
        config = config.get("text_config", config)
        return int(config["num_hidden_layers"])
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(name)
    config = getattr(config, "text_config", config)
    return int(config.num_hidden_layers)


def _uses_dar(config):
    model, run = (config["model"], config["run"])
    weaver = model.get("weaver", {})
    layers = [weaver.get("memory_layer", -1)]
    candidates = weaver.get("random_memory_layers")
    if isinstance(candidates, str):
        candidates = (
            []
            if candidates.strip().lower() in ("", "null", "none")
            else [value.strip() for value in candidates.split(",") if value.strip()]
        )
    candidates = list(candidates or [])
    actions = model.get("trigger_num_actions")
    actions = int(actions) if actions is not None else 3 if len(candidates) >= 2 else 2
    if run.get("mode") == "train" and run.get("train_weaver", False) and candidates:
        layers = candidates
    elif actions == 3 and len(candidates) >= 2:
        layers = candidates[:2]
    if all((str(layer).strip() == "-1" for layer in layers)):
        return False
    depth = _weaver_layer_count(model)
    for layer in layers:
        value = str(layer).strip().lower()
        if re.fullmatch("-?\\d+", value):
            index = int(value)
            if not -(depth + 1) <= index <= depth:
                raise ValueError(f"Invalid Weaver memory layer: {layer}")
            if index < 0:
                index += depth + 1
        elif value == "middle":
            index = 1 + depth // 2 if depth else 0
        else:
            match = re.fullmatch("l/(\\d+)(?:\\*(\\d+))?", value)
            if not match or int(match[1]) == 0:
                raise ValueError(f"Invalid Weaver memory layer: {layer}")
            fraction = int(match[2] or 1) / int(match[1])
            index = 1 + max(0, min(int(depth * fraction), depth - 1)) if depth else 0
        if index != depth:
            return True
    return False


def method_tags(config):
    dataset = config["dataset"]
    mode = dataset.get("mode", "sft")
    data_name = dataset.get(mode, {}).get("data_name", "official")
    if not isinstance(data_name, str) or not data_name.strip():
        raise ValueError("Run naming requires a nonempty dataset data_name")
    tags = []
    if data_name.strip() not in {"official", "new_official"}:
        tags.append("csmc")
    if _uses_dar(config):
        tags.append("dar")
    entropy = config["model"].get("entropy_trigger", {})
    mode = entropy.get("mode") or "none"
    if mode == "entropy_position":
        tags.append("rsat")
    return tags or ["memgen"]


def build_run_name(config, now=None):
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return (
        f"{'+'.join(method_tags(config))}_s{int(config['run'].get('seed', 42))}_{stamp}"
    )
