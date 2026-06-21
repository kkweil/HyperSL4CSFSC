from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from thop import profile


def _count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"params_total": total, "params_trainable": trainable, "buffers": buffers}


def _to_device(x: Any, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(i, device) for i in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def evaluate_model_compute_thop(
    model: nn.Module,
    example_inputs: Any,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
    flops_from_macs: bool = True,
) -> Dict[str, Any]:
    """
    使用 THOP 统计模型计算量（MACs + Params），并可转换为 FLOPs。

    Args:
        model: nn.Module
        example_inputs:
            - Tensor: model(x)
            - tuple/list: model(*inputs)  -> 这里建议用 tuple/list
            - dict: THOP profile 不支持 model(**kwargs) 这种通用方式；
                    如果你模型 forward 需要 kwargs，建议包装一个 Wrapper（见下方示例）。
        device: None 自动选择 cuda/CPU
        verbose: thop 的 verbose
        flops_from_macs: True 时 FLOPs = 2 * MACs（常见论文口径）

    Returns:
        dict: params_total, params_trainable, macs, flops, 以及可读字符串
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device).eval()
    inputs = _to_device(example_inputs, device)

    if torch.is_tensor(inputs):
        thop_inputs = (inputs,)
    elif isinstance(inputs, (tuple, list)):
        thop_inputs = tuple(inputs)
    elif isinstance(inputs, dict):
        raise TypeError(
            "THOP 的 profile 通常不直接支持 dict/kwargs 输入。"
            "请用 Wrapper 把 kwargs 变成位置参数，或改成 tuple 输入。"
        )
    else:
        thop_inputs = (inputs,)

    macs, thop_params = profile(model, inputs=thop_inputs, verbose=verbose)

    # THOP 返回的 params 是“参与计算的参数”，一般与总参数量一致，但以我们自己统计为准更稳。
    p = _count_params(model)
    flops = 2 * macs if flops_from_macs else macs

    def fmt(n: float, scale: float, unit: str) -> str:
        return f"{n/scale:.3f} {unit}"

    out = {
        **p,
        "macs": int(macs),
        "flops": int(flops),
        "thop_params": int(thop_params),
        "macs_str": fmt(macs, 1e9, "G"),
        "flops_str": fmt(flops, 1e9, "G"),
        "params_total_str": fmt(p["params_total"], 1e6, "M"),
        "params_trainable_str": fmt(p["params_trainable"], 1e6, "M"),
        "device": str(device),
        "flops_definition": "FLOPs = 2 * MACs" if flops_from_macs else "FLOPs = MACs",
    }
    return out
