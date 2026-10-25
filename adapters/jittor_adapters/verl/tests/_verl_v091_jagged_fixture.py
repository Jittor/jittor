# Exact function excerpts from verl 0.9.1 commit 1876b06d.
# Host-test imports are supplied by the fixture loader; not an implementation.

def nested_tensor_from_tensor_list(tensors: list[torch.Tensor], ragged_idx: int | None = None) -> torch.Tensor:
    assert len(tensors) > 0, "Must provide at least one tensor"
    sample_dim = tensors[0].dim()
    if ragged_idx is None:
        ragged_idx = sample_dim
    assert 1 <= ragged_idx <= sample_dim, (
        f"ragged_idx must be in [1, {sample_dim}]. Got {ragged_idx=} and {sample_dim=}"
    )

    if sample_dim == 1:
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    cat_dim = ragged_idx - 1
    values = torch.cat(tensors, dim=cat_dim)
    lengths = torch.tensor([tensor.shape[cat_dim] for tensor in tensors], dtype=torch.long, device=values.device)
    offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=values.device)
    torch.cumsum(lengths, dim=0, out=offsets[1:])

    nested_tensor = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
    nested_tensor._ragged_idx = ragged_idx
    return nested_tensor


def maybe_fix_3d_position_ids(data: TensorDict):
    # note for tensordict with pickle/unpickle. nested tensor in tensordict after consolidate and pickle/unpickle
    # will incur indexing error for ragged tensor. This only happens when using 3D position ids in VLMs.
    # This is likely a bug in tensordict. As a workaround, we manually set _ragged_index.
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        data["position_ids"]._ragged_idx = 2


def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert TensorDict from left-right padding to no-padding format.

    Args:
        data: TensorDict with "input_ids", "attention_mask", "response_mask", "position_ids"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors like "input_ids", "loss_mask", "position_ids"
        - NonTensorData includes "max_seq_len", "max_response_len", "indices"

    Note:
    1. the return input_ids/position_ids/loss_mask are nested tensor.
    2. we will remove "attention_mask", "response" in the return data, but "response_mask" is kept.
    """
    assert "input_ids" in data, "input_ids is required in left-right padding data"
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"
    assert "response_mask" in data, "response_mask is required in left-right padding data"
    assert "position_ids" in data, "position_ids is required in left-right padding data"

    input_ids = data.pop("input_ids")
    attention_mask = data["attention_mask"]
    response_mask = data["response_mask"]
    position_ids = data["position_ids"]  # (bs, seq_len) or # (bs, 4, seq_len)

    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    tu.assign_non_tensor_data(data, "indices", indices)

    input_ids_nested = torch.nested.nested_tensor_from_jagged(input_ids_rmpad.squeeze(-1), offsets=cu_seqlens)

    position_ids_list = []
    for i in range(attention_mask.shape[0]):
        curr_mask = attention_mask[i].bool()
        curr_pos_ids = position_ids[i]
        if curr_pos_ids.dim() == 1:  # (seq_len,)
            valid_ids = curr_pos_ids[curr_mask]
        else:  # (4, seq_len)
            valid_ids = curr_pos_ids[:, curr_mask]
        position_ids_list.append(valid_ids)
    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)

    data["input_ids"] = input_ids_nested
    data["position_ids"] = position_ids_nested
    data["loss_mask"] = data["response_mask"]

    routed_experts = data.get("routed_experts", None)
    if routed_experts is not None and not routed_experts.is_nested:
        routed_experts_rmpad = index_first_axis(routed_experts.unsqueeze(-1).flatten(0, 1), indices)
        routed_experts_nested = torch.nested.nested_tensor_from_jagged(
            routed_experts_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["routed_experts"] = routed_experts_nested

    # (bsz, seqlen, topk)
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None and teacher_ids is not None:
        teacher_logprobs_rmpad = index_first_axis(teacher_logprobs.unsqueeze(-1).flatten(0, 1), indices)
        teacher_ids_rmpad = index_first_axis(teacher_ids.unsqueeze(-1).flatten(0, 1), indices)
        teacher_logprobs_nested = torch.nested.nested_tensor_from_jagged(
            teacher_logprobs_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        teacher_ids_nested = torch.nested.nested_tensor_from_jagged(teacher_ids_rmpad.squeeze(-1), offsets=cu_seqlens)
        data["teacher_logprobs"] = teacher_logprobs_nested
        data["teacher_ids"] = teacher_ids_nested

    return data
