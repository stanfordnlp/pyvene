import torch

CONST_INPUT_HOOK = "register_forward_pre_hook"
CONST_OUTPUT_HOOK = "register_forward_hook"
CONST_GRAD_HOOK = "register_hook"


split_and_select = lambda x, num_slice, selct_index: torch.chunk(x, num_slice, dim=-1)[selct_index]
def split_heads(tensor, num_heads, attn_head_size):
    """Splits hidden_size dim into attn_head_size and num_heads."""
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

split_half = lambda x, selct_index: torch.chunk(x, 2, dim=-1)[selct_index]
split_three = lambda x, selct_index: torch.chunk(x, 3, dim=-1)[selct_index]
split_head_and_permute = lambda x, num_head: split_heads(x, num_head, x.shape[-1]//num_head)