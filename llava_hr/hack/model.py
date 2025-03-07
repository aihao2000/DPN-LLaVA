import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.utils import logging
import torch
from .utils import *

logger = logging.get_logger("transformers")


class PoolingRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(
                hidden_size,
            ),
            nn.Linear(
                hidden_size,
                hidden_size // 4,
                bias=False,
            ),
            nn.GELU(),
        )
        self.predictor = nn.Linear(
            hidden_size // 4,
            num_experts,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.mlp1[1].weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.mlp2[1].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.predictor.weight, a=math.sqrt(5))

    def forward(self, router_token):
        features = self.mlp1(router_token)
        # if vision_tokens is not None:
        #     features += self.mlp2(vision_tokens_mean)
        return self.predictor(features)

class SamplePoolingRouter(nn.Module):
    def __init__(self,hidden_size,num_experts):
        super().__init__()
        self.predictor = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.predictor.weight, a=math.sqrt(5))
    def forward(self, router_token):
        features = self.predictor(router_token)
        return features
class AdaptivePoolingLayer(nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size,
        kernel_size,
        stride,
        router_token_index,
        master_layer_index,
        log_routing_statistics,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.router = PoolingRouter(hidden_size, num_experts)
        self.experts = nn.ModuleDict()
        self.router_token_index = router_token_index
        self.master_layer_index = master_layer_index
        self.log_routing_statistics = log_routing_statistics
        for idx in range(num_experts - 1):
            self.experts[f"expert_{idx}"] = nn.Linear(
                hidden_size, hidden_size, bias=False
            )

        self.routing_statistics = {}
        self.routing_statistics["router_count"] = {}
        self.routing_statistics["logits"] = []
        for i in range(len(kernel_size)):
            self.routing_statistics["router_count"][i] = 0

    def forward(self, hidden_states, labels, input_ids, vision_token_mask):
        (
            num_vision_tokens,
            vision_token_min_indices,
            vision_token_max_indices,
        ) = parse_vision_token_mask(vision_token_mask)
        router_states = []
        new_hidden_states = []
        for (
            single_hidden_states,
            single_input_ids,
            single_labels,
            single_vision_token_mask,
            single_num_vision_tokens,
        ) in zip(
            hidden_states,
            input_ids,
            labels,
            vision_token_mask,
            num_vision_tokens,
        ):
            single_hidden_states = single_hidden_states.view(
                (1, hidden_states.shape[1], hidden_states.shape[2])
            ).clone()
            router_token_indices = (
                torch.nonzero(
                    (single_input_ids == self.router_token_index),
                    as_tuple=True,
                )[0]
                + single_num_vision_tokens
                - 1
            )
            router_tokens = single_hidden_states[:, router_token_indices]
            cls_token = router_tokens.mean(dim=1)
            logits = self.router(router_token=cls_token) / 10  # 1,3
            logits = F.softmax(logits, dim=-1).to(
                hidden_states.device, hidden_states.dtype
            )
            if self.training:
                index = int(torch.multinomial(logits, num_samples=1).item())
            else:
                index = torch.max(
                    logits,
                    dim=-1,
                )
                if self.log_routing_statistics:
                    self.routing_statistics["router_count"][index] += 1
                    # self.routing_statistics["cls_tokens"].append(cls_token)
                    # self.routing_statistics["logits"].append(logits.detach().to("cpu"))
            router_states.append((logits, index, self.master_layer_index))
            if index != 0:
                single_hidden_states[:, router_token_indices] = self.experts[
                    f"expert_{index-1}"
                ](router_tokens)
            new_hidden_states.append(single_hidden_states)
        hidden_states = torch.concat(new_hidden_states, dim=0)

        new_hidden_states = []
        new_vision_token_mask = []
        for (
            single_hidden_states,
            single_input_ids,
            single_labels,
            single_vision_token_mask,
            single_num_vision_tokens,
            min_vision_token_index,
            max_vision_token_index,
            single_router_state,
        ) in zip(
            hidden_states,
            input_ids,
            labels,
            vision_token_mask,
            num_vision_tokens,
            vision_token_min_indices,
            vision_token_max_indices,
            router_states,
        ):
            logits, index, master_layer_index = single_router_state
            probs = logits[0, index]

            single_hidden_states = single_hidden_states.view(
                (1, hidden_states.shape[1], hidden_states.shape[2])
            )
            single_vision_tokens = single_hidden_states[
                :, min_vision_token_index : max_vision_token_index + 1, :
            ]
            new_single_vision_token_mask = single_vision_token_mask.clone()
            if (
                single_num_vision_tokens == 0
                or single_num_vision_tokens < 16
                or index == 0
            ):
                new_hidden_states.append(single_hidden_states)
                new_vision_token_mask.append(new_single_vision_token_mask)
                continue

            height = width = int(math.sqrt(single_vision_tokens.shape[1]))

            if any(
                [kernel_size[0] != kernel_size[1] for kernel_size in self.kernel_size]
            ):
                if height * width != single_vision_tokens.shape[1]:
                    height = int(math.sqrt(single_vision_tokens.shape[1] / 2))
                    width = height * 2

            single_vision_tokens = single_vision_tokens.view(
                (1, height, width, hidden_states.shape[-1])
            )
            single_vision_tokens = single_vision_tokens.permute(0, 3, 1, 2)  # b,c,h,w

            kernel_size = self.kernel_size[index]
            stride = self.stride[index]
            single_vision_tokens = F.max_pool2d(
                single_vision_tokens,
                kernel_size=kernel_size,
                stride=stride,
            )
            single_vision_tokens = probs * single_vision_tokens

            single_vision_tokens = single_vision_tokens.permute(0, 2, 3, 1)  # b,h,w,c
            single_vision_tokens = single_vision_tokens.view(
                (1, -1, hidden_states.shape[-1])
            )

            new_num_vision_tokens = single_vision_tokens.shape[1]
            new_single_hidden_states = torch.concat(
                [
                    single_hidden_states[:, :min_vision_token_index, :],
                    single_vision_tokens,
                    single_hidden_states[:, max_vision_token_index + 1 :],
                ],
                dim=1,
            )
            num_padding_tokens = (
                single_hidden_states.shape[1] - new_single_hidden_states.shape[1]
            )
            if num_padding_tokens > 0 and self.training:
                new_single_hidden_states = torch.concat(
                    [
                        new_single_hidden_states,
                        torch.zeros(
                            (
                                1,
                                num_padding_tokens,
                                hidden_states.shape[-1],
                            ),
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        ),
                    ],
                    dim=1,
                )
            new_hidden_states.append(new_single_hidden_states)

            new_single_vision_token_mask[
                min_vision_token_index + new_num_vision_tokens :
            ] = False
            new_vision_token_mask.append(new_single_vision_token_mask)

        new_hidden_states = torch.concat(new_hidden_states, dim=0)
        return (new_hidden_states, new_vision_token_mask, router_states)
