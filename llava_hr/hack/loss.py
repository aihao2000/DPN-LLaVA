import torch
from torch.nn import CrossEntropyLoss
import math
from transformers.utils import logging

logger = logging.get_logger("transformers")


def calculate_hinge_loss(pooling_layers, router_states, device, dtype):

    pooling_layer_indices = [single_layer["index"] for single_layer in pooling_layers]
    pooling_route_loss = [[]] * len(pooling_layers)
    scales = torch.tensor([0, 1, 2], dtype=dtype, device=device)
    for (
        single_router_logits,
        single_router_labels,
        layer_idx,
    ) in router_states:
        single_router_logits = single_router_logits.view((-1,))
        max_index = torch.argmax(single_router_logits)
        pooling_route_loss[pooling_layer_indices.index(layer_idx)].append(
            (scales[max_index] * single_router_logits[max_index])
        )
    pooling_route_loss = [
        torch.stack(single_layer_pooling_route_loss).mean()
        for single_layer_pooling_route_loss in pooling_route_loss
    ]
    pooling_route_loss = [
        max(
            torch.tensor([0], device=device, dtype=dtype),
            1.5 - single_layer_pooling_route_loss,
        )
        for single_layer_pooling_route_loss in pooling_route_loss
    ]
    pooling_route_loss = torch.stack(pooling_route_loss).mean()
    return pooling_route_loss