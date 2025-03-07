import torch


def parse_vision_token_mask(vision_token_mask):
    vision_token_min_indices = []
    vision_token_max_indices = []
    num_vision_tokens = []
    for single_vision_token_mask in vision_token_mask:
        nonzero_indices = torch.nonzero(single_vision_token_mask, as_tuple=True)[0]
        if len(nonzero_indices) == 0:
            min_vision_token_index = -1
            max_vision_token_index = -2
        else:
            min_vision_token_index = nonzero_indices[0].item()
            max_vision_token_index = nonzero_indices[-1].item()

        vision_token_min_indices.append(min_vision_token_index)
        vision_token_max_indices.append(max_vision_token_index)

        num_vision_tokens.append(max_vision_token_index - min_vision_token_index + 1)
    return num_vision_tokens, vision_token_min_indices, vision_token_max_indices


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )
