from .forward import *
from .model import *
from transformers import set_seed
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

import random
import json
import os

logger = logging.get_logger("transformers")


def hack_llava(
    llava,
    tokenizer=None,
    pooling_type="adaptive",
    pooling_location="post",
    pooling_layers=[
        {
            "index": 7,
            "kernel_size": [(1, 1), (1, 2), (2, 2)],
            "stride": [(1, 1), (1, 2), (2, 2)],
        },
        {
            "index": 15,
            "kernel_size": [(1, 1), (1, 2), (2, 2)],
            "stride": [(1, 1), (1, 2), (2, 2)],
        },
        {
            "index": 23,
            "kernel_size": [(1, 1), (1, 2), (2, 2)],
            "stride": [(1, 1), (1, 2), (2, 2)],
        },
    ],
    pooling_function="max",
    prepooling=False,
    router_version="v1",
    weighting_vision_tokens=False,
    ckpt_path=None,
    use_special_token=False,
    log_routing_statistics=True,
    split_autoregressive_loss=False,
    use_router_contrastive_loss=False,
    detach_cls_token_to_predict=False,
):
    assert pooling_type in ["static", "adaptive", "random"]
    assert pooling_function in ["max", "avg", "conv"]
    assert pooling_location in ["pre", "post"]

    if ckpt_path is None:
        state_dict = None
    else:
        state_dict = {}
        ckpt_paths = [
            f"{ckpt_path}/{path}"
            for path in os.listdir(ckpt_path)
            if path.startswith("pytorch_model-")
        ]
        for single_ckpt_path in ckpt_paths:
            single_state_dict = torch.load(single_ckpt_path)
            for key in single_state_dict.keys():
                if "pooling_router" in key:
                    state_dict[key] = single_state_dict[key]
                if "experts" in key:
                    state_dict[key] = single_state_dict[key]
        with open(f"{ckpt_path}/config.json") as f:
            config = json.load(f)
            if "pooling" in config.keys():
                pooling_type = config["pooling"]["pooling_type"]
                pooling_layers = config["pooling"]["pooling_layers"]
                use_special_token = config["pooling"]["use_special_token"]
                if "pooling_function" in config["pooling"].keys():
                    pooling_function = config["pooling"]["pooling_function"]

                if "pooling_location" in config["pooling"].keys():
                    pooling_location = config["pooling"]["pooling_location"]
                if "router_version" in config["pooling"].keys():
                    router_version = config["pooling"]["router_version"]
                if "prepooling" in config["pooling"].keys():
                    prepooling = config["pooling"]["prepooling"]
            else:
                pooling_type = "static"
                pooling_layers = []
    set_seed(42)
    if use_special_token:
        num_add_tokens = tokenizer.add_tokens(["<ROUTER>"], special_tokens=True)
        if num_add_tokens == 0:
            logger.warn("<ROUTER> exists in tokenizer")
        else:
            llava.resize_token_embeddings(len(tokenizer))
        router_token_index = tokenizer.encode("<ROUTER>")[1]
        llava.router_token_index = router_token_index

    llava.pooling_layers = pooling_layers
    llava.split_autoregressive_loss = split_autoregressive_loss
    llava.use_router_contrastive_loss = use_router_contrastive_loss
    llava.get_vision_tower().prepooling = prepooling
    pooling_layer_indices = [pooling_layer["index"] for pooling_layer in pooling_layers]
    llava.log_routing_statistics = log_routing_statistics
    if log_routing_statistics:
        llava.routing_statistics = {}
    for i, layer in enumerate(llava.model.layers):
        layer.forward = hacked_llama_decoder_layer_forward(layer)
        layer.layer_idx = i
        if i in pooling_layer_indices:
            kernel_size = pooling_layers[pooling_layer_indices.index(i)]["kernel_size"]
            stride = pooling_layers[pooling_layer_indices.index(i)]["stride"]
            layer.pooling_type = pooling_type
            layer.pooling_parameter = {
                "kernel_size": kernel_size,
                "stride": stride,
            }
            layer.pooling_function = pooling_function
            layer.pooling_location = pooling_location

            layer.use_special_token = use_special_token
            layer.split_autoregressive_loss = split_autoregressive_loss

            layer.weighting_vision_tokens = weighting_vision_tokens
            layer.detach_cls_token_to_predict = detach_cls_token_to_predict

            if pooling_type == "adaptive":
                if router_version == "v1":
                    layer.pooling_router = PoolingRouter(
                        hidden_size=layer.self_attn.q_proj.in_features,
                        num_experts=len(kernel_size),
                    ).to(llava.device, llava.dtype)
                elif router_version == "v2":
                    layer.pooling_router = SamplePoolingRouter(
                        hidden_size=layer.self_attn.q_proj.in_features,
                        num_experts=len(kernel_size),
                    ).to(llava.device, llava.dtype)
            if use_special_token:
                layer.router_token_index = router_token_index
            if pooling_function == "conv":
                experts = nn.Sequential()
                for i, (single_kernel_size, single_stride) in enumerate(
                    zip(kernel_size, stride)
                ):
                    conv = nn.Conv2d(
                        layer.self_attn.q_proj.in_features,
                        layer.self_attn.q_proj.in_features,
                        kernel_size=single_kernel_size,
                        stride=single_stride,
                        bias=False,
                        groups=layer.self_attn.q_proj.in_features,
                    )
                    # nn.init.constant_(conv.bias, 0)
                    nn.init.constant_(
                        conv.weight, 1 / (single_kernel_size[0] * single_kernel_size[1])
                    )
                    experts.add_module(str(i), conv)
                layer.experts = experts.to(llava.device, llava.dtype)
    if state_dict is not None:
        current_model_dict = llava.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if current_model_dict[k].shape == v.shape
        }
        logger.warn(f"load state dict: {state_dict.keys()}")
        llava.load_state_dict(state_dict, strict=False)
    llava.model.forward = hacked_llama_base_model_forward(llava.model)

    llava.config.pooling = {
        "pooling_type": pooling_type,
        "pooling_layers": pooling_layers,
        "pooling_function": pooling_function,
        "pooling_location": pooling_location,
        "prepooling": prepooling,
        "use_special_token": use_special_token,
        "detach_cls_token_to_predict": detach_cls_token_to_predict,
        "router_version": router_version,
    }
    logger.warn(f"pooling_config: {llava.config.pooling}")


def log_routing_statistics(llava):
    if hasattr(llava, "routing_statistics"):
        logger.warn("routing_statistics:\n")
        logger.warning(
            f"first_token_inference_time:{sum(llava.first_token_inference_time)/len(llava.first_token_inference_time)}"
        )
        for k in sorted(llava.routing_statistics.keys()):
            logger.warn((k, llava.routing_statistics[k]))
    # logger.warning(f"avg inference speed: {sum(llava.inference_speed) / len(llava.inference_speed)}")
