#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.utils import logging
from ...hack.vision_token_purning_handler import hack_llava
from ...hack.loss import *
from ...hack.utils import parse_vision_token_mask
import time

logger = logging.get_logger("transformers")


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


MAX_NUM_VISON_TOKENS = 1024


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # HACK
        prepared_inputs = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        if len(prepared_inputs) == 6:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_token_mask,
            ) = prepared_inputs
        else:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = prepared_inputs
            vision_token_mask = None
            # logger.warning("vision token mask is None!!!")

        if vision_token_mask is not None:
            (
                num_vision_tokens,
                vision_token_min_indices,
                vision_token_max_indices,
            ) = parse_vision_token_mask(vision_token_mask)
            vision_map_shapes = [
                (
                    int(math.sqrt(single_num_vision_tokens)),
                    int(math.sqrt(single_num_vision_tokens)),
                )
                for single_num_vision_tokens in num_vision_tokens
            ]
            if num_vision_tokens[0] != 0:
                self.max_num_vision_tokens = num_vision_tokens[0]
                for layer in self.model.layers:
                    layer.max_num_vision_tokens = num_vision_tokens[0]
        else:
            vision_map_shapes = None

        if (
            hasattr(self.config, "pooling")
            and self.config.pooling["use_special_token"]
            and self.training
        ):
            new_inputs_embeds = []
            router_token_embeds = self.get_model().embed_tokens(
                torch.tensor([self.router_token_index]).to(
                    self.device, dtype=input_ids.dtype
                )
            )
            for single_input_embeds, single_input_ids in zip(inputs_embeds, input_ids):
                router_token_indices = (
                    torch.nonzero(
                        single_input_ids == self.router_token_index, as_tuple=True
                    )[0]
                    + self.max_num_vision_tokens
                    - 1
                )
                for router_token_index in router_token_indices:
                    single_input_embeds = torch.concat(
                        [
                            single_input_embeds[:router_token_index],
                            router_token_embeds,
                            single_input_embeds[router_token_index + 1 :],
                        ]
                    )
                new_inputs_embeds.append(single_input_embeds)
            inputs_embeds = torch.stack(new_inputs_embeds)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        start_time = time.time()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # HACK
            vision_token_mask=vision_token_mask,
            labels=labels,
            vision_map_shapes=vision_map_shapes,
        )
        end_time = time.time()
        # HACK
        if isinstance(outputs, tuple):
            (outputs, new_vision_token_mask, router_states) = outputs
        else:
            new_vision_token_mask = None
            router_states = None
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if new_vision_token_mask is not None:
                new_labels = []
                for single_labels, single_vision_token_mask in zip(
                    labels, new_vision_token_mask
                ):
                    nonzero_indices = torch.nonzero(
                        single_vision_token_mask, as_tuple=True
                    )[0]
                    if len(nonzero_indices) == 0:
                        new_labels.append(single_labels)
                    else:
                        min_vision_token_index = nonzero_indices[0].item()
                        max_vision_token_index = nonzero_indices[-1].item()
                        num_vision_tokens = (
                            max_vision_token_index - min_vision_token_index + 1
                        )
                        single_labels = torch.concat(
                            [
                                single_labels[:min_vision_token_index],
                                single_labels[
                                    min_vision_token_index : max_vision_token_index + 1,
                                ],
                                single_labels[
                                    min_vision_token_index
                                    + self.max_num_vision_tokens :
                                ],
                                torch.full(
                                    (self.max_num_vision_tokens - num_vision_tokens,),
                                    -100,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                ),
                            ],
                        )
                        new_labels.append(single_labels)

                labels = torch.stack(new_labels)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if len(router_states) > 0:
                hinge_loss = calculate_hinge_loss(
                    pooling_layers=self.pooling_layers,
                    router_states=router_states,
                    device=loss.device,
                    dtype=loss.dtype,
                )
                logger.warn(f"ori loss: {loss} hinge_loss: {hinge_loss}")
                # scale = 0.5 * math.exp(-abs(loss.item()))
                loss = loss + 0.001 * hinge_loss

        if self.log_routing_statistics:
            if not hasattr(self, "first_token_inference_time"):
                self.first_token_inference_time = []
            self.first_token_inference_time.append(end_time - start_time)
            if len(router_states) > 0:
                routing_path = str(
                    [
                        self.model.layers[layer_idx].pooling_parameter["kernel_size"][
                            single_router_labels
                        ][0]
                        * self.model.layers[layer_idx].pooling_parameter["kernel_size"][
                            single_router_labels
                        ][1]
                        for single_router_logits, single_router_labels, layer_idx in router_states
                    ]
                )
                if routing_path not in self.routing_statistics.keys():
                    self.routing_statistics[routing_path] = 0
                self.routing_statistics[routing_path] += 1
                self.routing_path = routing_path

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
