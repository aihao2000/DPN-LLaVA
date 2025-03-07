from transformers.models.llama.modeling_llama import *
from transformers import LlamaForCausalLM
from transformers import set_seed
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

import random
import json
from deepspeed.moe.layer import MoE
from .utils import *

logger = logging.get_logger("transformers")



def hacked_llama_decoder_layer_forward(self):
    def forward(
        # self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        labels=None,
        input_ids=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        # HACK
        vision_token_mask=None,
        vision_map_shapes=None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        # HACK
        router_states = []
        new_vision_map_shapes = vision_map_shapes
        if vision_token_mask is not None:
            (
                num_vision_tokens,
                vision_token_min_indices,
                vision_token_max_indices,
            ) = parse_vision_token_mask(vision_token_mask=vision_token_mask)
        if (
            past_key_value is None
            and vision_token_mask is not None
            and max(num_vision_tokens) >= 4
            and hasattr(self, "pooling_type")
            and len(set(num_vision_tokens)) != 1
            and 0 in set(num_vision_tokens)
            and self.training
        ):
            logger.warn(
                f"Inconsistent batches when forward: num_vision_tokens: {num_vision_tokens} "
            )
        if hasattr(self, "pooling_parameter"):
            if isinstance(self.pooling_parameter["stride"][0], int):
                max_stride = (
                    self.pooling_parameter["stride"][0]
                    * self.pooling_parameter["stride"][1]
                )
            else:
                max_stride = max(
                    stride[0] * stride[1] for stride in self.pooling_parameter["stride"]
                )

        if (
            past_key_value is None
            and vision_token_mask is not None
            and hasattr(self, "pooling_type")
            and all(
                single_num_vision_tokens >= max_stride
                for single_num_vision_tokens in num_vision_tokens
            )
            and self.pooling_location == "pre"
        ):
            if labels is None:
                labels = [None] * hidden_states.shape[0]
            (
                hidden_states,
                new_vision_token_mask,
                router_states,
                new_vision_map_shapes,
            ) = unbatch_pooling(
                layer=self,
                hidden_states=hidden_states,
                vision_token_mask=vision_token_mask,
                labels=labels,
                input_ids=input_ids,
                vision_map_shapes=vision_map_shapes,
            )
        else:
            new_vision_token_mask = vision_token_mask

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if not self.training:
            if past_key_value is not None:
                past_key_value_length = past_key_value[0].shape[2]
                position_ids = torch.arange(
                    past_key_value_length,
                    hidden_states.shape[1] + past_key_value_length,
                    dtype=torch.long,
                    device=hidden_states.device,
                )

                attention_mask = torch.zeros(
                    (
                        hidden_states.shape[0],
                        1,
                        hidden_states.shape[1],
                        past_key_value_length + 1,
                    ),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            else:
                position_ids = torch.arange(
                    0,
                    hidden_states.shape[1],
                    dtype=torch.long,
                    device=hidden_states.device,
                ).view((1, hidden_states.shape[1]))
                attention_mask = make_causal_mask(
                    position_ids.shape,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
        # Self Attentione:
        (
            hidden_states,
            self_attn_weights,
            present_key_value,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # UNBATCH SELF ATTN

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if (
            past_key_value is None
            and vision_token_mask is not None
            and hasattr(self, "pooling_type")
            and all(
                single_num_vision_tokens >= max_stride
                for single_num_vision_tokens in num_vision_tokens
            )
            and self.pooling_location == "post"
        ):
            if labels is None:
                labels = [None] * hidden_states.shape[0]
            (
                hidden_states,
                new_vision_token_mask,
                router_states,
                new_vision_map_shapes,
            ) = unbatch_pooling(
                layer=self,
                hidden_states=hidden_states,
                vision_token_mask=vision_token_mask,
                labels=labels,
                input_ids=input_ids,
                vision_map_shapes=vision_map_shapes,
            )
        else:
            new_vision_token_mask = vision_token_mask

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # HACK
        outputs += (new_vision_map_shapes,)
        outputs += (new_vision_token_mask,)

        outputs += (router_states,)
        return outputs

    return forward


def hacked_llama_base_model_forward(self):
    def forward(
        # self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # HACK
        vision_token_mask=None,
        labels=None,
        vision_map_shapes=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            # raise ValueError(
            #     "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            # )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        router_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(
                    module, vision_token_mask_, vision_map_shapes_
                ):
                    VISION_TOKEN_MASK = vision_token_mask_
                    VISION_MAP_SHAPES = vision_map_shapes_

                    # hidden_states: torch.Tensor,
                    # attention_mask: Optional[torch.Tensor] = None,
                    # position_ids: Optional[torch.LongTensor] = None,
                    # past_key_value: Optional[Tuple[torch.Tensor]] = None,
                    # output_attentions: Optional[bool] = False,
                    # use_cache: Optional[bool] = False,
                    # # HACK
                    # vision_token_mask=None,
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module.forward(
                            *inputs,
                            output_attentions,
                            None,
                            VISION_TOKEN_MASK,
                            VISION_MAP_SHAPES,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(
                        decoder_layer, vision_token_mask, vision_map_shapes
                    ),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    labels,
                    input_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # HACK
                    vision_token_mask=vision_token_mask,
                    labels=labels,
                    input_ids=input_ids,
                    vision_map_shapes=vision_map_shapes,
                )

            # HACK
            vision_map_shapes = layer_outputs[-3]
            vision_token_mask = layer_outputs[-2]
            router_states += layer_outputs[-1]

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ),
            vision_token_mask,
            router_states,
        )

    return forward


def unbatch_pooling(
    layer, hidden_states, labels, input_ids, vision_token_mask, vision_map_shapes
):
    (
        num_vision_tokens,
        vision_token_min_indices,
        vision_token_max_indices,
    ) = parse_vision_token_mask(vision_token_mask)
    new_hidden_states = []
    new_vision_token_mask = []
    new_vision_map_shapes = []
    router_states = []
    batch_size = hidden_states.shape[0]
    if layer.pooling_function == "conv":
        used_experts = set()
    for i, (
        single_hidden_states,
        single_input_ids,
        single_labels,
        single_vision_token_mask,
        single_num_vision_tokens,
        min_vision_token_index,
        max_vision_token_index,
        (height, width),
    ) in enumerate(
        zip(
            hidden_states,
            input_ids,
            labels,
            vision_token_mask,
            num_vision_tokens,
            vision_token_min_indices,
            vision_token_max_indices,
            vision_map_shapes,
        )
    ):
        single_hidden_states = single_hidden_states.view(
            (1, hidden_states.shape[1], hidden_states.shape[2])
        )
        single_vision_tokens = single_hidden_states[
            :, min_vision_token_index : max_vision_token_index + 1, :
        ]
        new_single_vision_token_mask = single_vision_token_mask.clone()
        if single_num_vision_tokens == 0 or single_num_vision_tokens < 16:
            new_hidden_states.append(single_hidden_states)
            new_vision_token_mask.append(new_single_vision_token_mask)
            continue
        if layer.pooling_type == "adaptive":
            if hasattr(layer, "use_special_token") and layer.use_special_token:
                router_token_indices = (
                    torch.nonzero(
                        (single_input_ids == layer.router_token_index), as_tuple=True
                    )[0]
                    + single_num_vision_tokens
                    - 1
                )
                if router_token_indices.shape[0] > 0:
                    cls_token = single_hidden_states[:, router_token_indices].mean(
                        dim=1
                    )
                    if layer.weighting_vision_tokens:
                        raise ValueError("weighting vision token not implementation")
                        # weights = (
                        #     F.cosine_similarity(cls_token, single_vision_tokens[0]) + 1
                        # ) / 2
                        # weights = weights.view(
                        #     (1, single_vision_tokens.shape[1], 1)
                        # ).repeat((1, 1, single_vision_tokens.shape[2]))
                        # single_vision_tokens = single_vision_tokens * weights
                    if layer.detach_cls_token_to_predict:
                        cls_token = cls_token.detach()
                else:
                    cls_token = None
            else:
                if single_labels is None:
                    cls_token = single_hidden_states[:, -1]
                else:
                    single_labels = torch.concat(
                        [
                            single_labels[:min_vision_token_index],
                            single_labels[
                                min_vision_token_index : max_vision_token_index + 1,
                            ],
                            single_labels[
                                min_vision_token_index + layer.max_num_vision_tokens :
                            ],
                        ],
                    )
                    last_index = torch.nonzero(single_labels >= 0)
                    if len(last_index) > 0:
                        last_index = last_index[0].item() - 1
                        cls_token = single_hidden_states[:, last_index]
                    else:
                        cls_token = single_hidden_states[:, -1]

        if layer.pooling_function == "max":
            pooling_function = F.max_pool2d
        elif layer.pooling_function == "avg":
            pooling_function = F.avg_pool2d

        # if all(
        #     kernel_size[0] == 1
        #     for kernel_size in layer.pooling_parameter["kernel_size"]
        # ):
        #     height = 1
        #     width = single_num_vision_tokens
        # height = width = int(math.sqrt(single_vision_tokens.shape[1]))
        # if layer.pooling_type == "adaptive" and any(
        #     [
        #         kernel_size[0] != kernel_size[1]
        #         for kernel_size in layer.pooling_parameter["kernel_size"]
        #     ]
        # ):
        #     if height * width != single_vision_tokens.shape[1]:
        #         height = int(math.sqrt(single_vision_tokens.shape[1] / 2))
        #         width = height * 2

        single_vision_tokens = single_vision_tokens.view(
            (1, height, width, hidden_states.shape[-1])
        )
        single_vision_tokens = single_vision_tokens.permute(0, 3, 1, 2)  # b,c,h,w

        if layer.pooling_type == "static":
            single_vision_tokens = pooling_function(
                single_vision_tokens,
                kernel_size=layer.pooling_parameter["kernel_size"],
                stride=layer.pooling_parameter["stride"],
            )
        elif layer.pooling_type == "random":
            index = random.randint(0, len(layer.pooling_parameter["kernel_size"]) - 1)
            kernel_size = layer.pooling_parameter["kernel_size"][index]
            stride = layer.pooling_parameter["stride"][index]
            single_vision_tokens = pooling_function(
                single_vision_tokens,
                kernel_size=kernel_size,
                stride=stride,
            )
        elif layer.pooling_type == "adaptive" and cls_token is not None:
            logits = layer.pooling_router(router_token=cls_token) / 10  # 1,3
            # logits = logits + 0.1 * torch.randn_like(logits)
            # logger.warning(logits)
            logits = F.softmax(logits, dim=-1).to(
                hidden_states.device, hidden_states.dtype
            )

            if layer.training:
                index = int(torch.multinomial(logits, num_samples=1).item())
                new_single_num_vision_tokens = int(
                    single_num_vision_tokens
                    / (
                        layer.pooling_parameter["stride"][index][0]
                        * layer.pooling_parameter["stride"][index][1]
                    )
                )
                while new_single_num_vision_tokens < 18:
                    index = int(torch.multinomial(logits, num_samples=1).item())
                    new_single_num_vision_tokens = int(
                        single_num_vision_tokens
                        / (
                            layer.pooling_parameter["stride"][index][0]
                            * layer.pooling_parameter["stride"][index][1]
                        )
                    )

                prob = logits[0, index]
                # probs, index = torch.max(
                #     logits,
                #     dim=-1,
                # )
                # prob = probs[0]
                if layer.pooling_function == "conv":
                    if (batch_size - i) <= len(layer.experts) - len(used_experts):
                        while index in used_experts:
                            index = int(torch.multinomial(logits, num_samples=1).item())
                            prob = logits[0, index]
                    single_vision_tokens = layer.experts[index](
                        single_vision_tokens,
                    )
                    used_experts.add(index)
                else:
                    if (
                        layer.pooling_parameter["stride"][index][0] != 1
                        or layer.pooling_parameter["stride"][index][1] != 1
                    ):
                        kernel_size = layer.pooling_parameter["kernel_size"][index]
                        stride = layer.pooling_parameter["stride"][index]
                        single_vision_tokens = pooling_function(
                            single_vision_tokens,
                            kernel_size=kernel_size,
                            stride=stride,
                        )
                single_vision_tokens = prob * single_vision_tokens
                router_states.append((logits, index, layer.layer_idx))
            else:
                probs, index = torch.max(
                    logits,
                    dim=-1,
                )
                prob = probs[0]
                index = int(index.item())

                if layer.pooling_function == "conv":
                    single_vision_tokens = layer.experts[index](
                        single_vision_tokens,
                    )
                else:
                    if (
                        layer.pooling_parameter["stride"][index][0] != 1
                        or layer.pooling_parameter["stride"][index][1] != 1
                    ):
                        kernel_size = layer.pooling_parameter["kernel_size"][index]
                        stride = layer.pooling_parameter["stride"][index]
                        single_vision_tokens = pooling_function(
                            single_vision_tokens,
                            kernel_size=kernel_size,
                            stride=stride,
                        )
                single_vision_tokens = single_vision_tokens * prob
                router_states.append((logits, index, layer.layer_idx))
        else:
            raise ValueError(f"pooling type {layer.pooling_type} error")

        single_vision_tokens = single_vision_tokens.permute(0, 2, 3, 1)  # b,h,w,c
        new_vision_map_shapes.append(
            (single_vision_tokens.shape[1], single_vision_tokens.shape[2])
        )
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
        if num_padding_tokens > 0 and layer.training:
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

    return (
        new_hidden_states,
        new_vision_token_mask,
        router_states,
        new_vision_map_shapes,
    )


# def unbatch_self_attn(
#     self_attn,
#     hidden_states,
#     attention_mask,
#     position_ids,
#     vision_token_mask,
#     output_attentions,
#     use_cache,
# ):
#     new_hidden_states = []
#     is_flash_attention = len(attention_mask.shape) == 2
#     (
#         num_vision_tokens,
#         vision_token_min_indices,
#         vision_token_max_indices,
#     ) = parse_vision_token_mask(vision_token_mask)
#     for (
#         single_hidden_states,
#         single_attention_mask,
#         single_num_vision_tokens,
#         min_vision_token_index,
#         max_vision_token_index,
#     ) in zip(
#         hidden_states,
#         attention_mask,
#         num_vision_tokens,
#         vision_token_min_indices,
#         vision_token_max_indices,
#     ):
#         single_hidden_states = single_hidden_states.view(
#             (1, hidden_states.shape[1], hidden_states.shape[2])
#         )
#         single_attention_mask = single_attention_mask.view(
#             (1, *single_attention_mask.shape)
#         )
#         if single_num_vision_tokens == 0:
#             (
#                 new_single_hidden_states,
#                 single_self_attn_weights,
#                 single_present_key_value,
#             ) = self_attn(
#                 hidden_states=single_hidden_states,
#                 attention_mask=single_attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=None,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )
#         else:
#             num_padding_tokens = self.max_num_vision_tokens - single_num_vision_tokens
#             if num_padding_tokens > 0:
#                 if is_flash_attention:
#                     # flash attention

#                     single_attention_mask = single_attention_mask[
#                         :, : single_attention_mask.shape[1] - num_padding_tokens
#                     ]
#                 else:
#                     single_attention_mask = make_causal_mask(
#                         (1, single_hidden_states.shape[1]),
#                         dtype=hidden_states.dtype,
#                         device=hidden_states.device,
#                         past_key_values_length=0,
#                     )

#             # logger.warn(
#             #     f"position_ids.shape:{position_ids.shape}\n single_hidden_states.shape:{single_hidden_states.shape}"
#             # )
#             (
#                 new_single_hidden_states,
#                 single_self_attn_weights,
#                 single_present_key_value,
#             ) = self_attn(
#                 hidden_states=single_hidden_states[
#                     :, : single_hidden_states.shape[1] - num_padding_tokens
#                 ],
#                 attention_mask=single_attention_mask,
#                 position_ids=position_ids[
#                     :, : single_hidden_states.shape[1] - num_padding_tokens
#                 ],
#                 past_key_value=None,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )
#             if num_padding_tokens > 0:
#                 new_single_hidden_states = torch.concat(
#                     [
#                         new_single_hidden_states,
#                         single_hidden_states[
#                             :, single_hidden_states.shape[1] - num_padding_tokens :
#                         ],
#                     ],
#                     dim=1,
#                 )

#         new_hidden_states.append(new_single_hidden_states)

#     new_hidden_states = torch.concat(new_hidden_states, dim=0)
#     return new_hidden_states, None, None
