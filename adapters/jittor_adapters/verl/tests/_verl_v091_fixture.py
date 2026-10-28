# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exact six-function fixture from verl 0.9.1, commit 1876b06d.
# Host tests supply imports; not an implementation.

def concat_tensordict(data: list[TensorDict]) -> TensorDict:
    """Concatenate multiple TensorDicts along dimension zero.

    Combines a list of TensorDicts into a single TensorDict by concatenating
    all tensors along the batch dimension (dim=0). Handles nested tensors
    specially by unbinding and rebinding them.

    Args:
        data: List of TensorDicts to concatenate. All TensorDicts must have
            the same keys and the same set of nested tensor keys.

    Returns:
        A new TensorDict containing concatenated tensors from all inputs.

    Raises:
        AssertionError: If data is empty or if TensorDicts have inconsistent
            nested tensor keys.

    Note:
        - For TensorDicts with empty batch_size, returns the first one
        - Nested tensors are handled specially via concat_nested_tensors
        - Regular tensors use TensorDict.cat for efficient concatenation
    """
    assert len(data) > 0, "Must have at least one tensordict"

    # Find nested tensor keys from the first tensordict
    nested_tensor_keys = {key for key, value in data[0].items() if isinstance(value, torch.Tensor) and value.is_nested}

    if not nested_tensor_keys:
        if len(data[0].batch_size) == 0:
            return concat_tensordict_with_none_bsz(data)
        # if batch size is None (only contain NonTensorData)
        return TensorDict.cat(data, dim=0)

    # Create a list of tensordicts containing only non-nested tensors for concatenation
    regular_tds = []
    for td in data:
        current_nested_keys = {k for k, v in td.items() if isinstance(v, torch.Tensor) and v.is_nested}
        assert current_nested_keys == nested_tensor_keys, "All tensordicts must have the same set of nested tensors."

        # Create a new TensorDict with non-nested items without modifying the original
        regular_items = {k: v for k, v in td.items() if k not in nested_tensor_keys}
        regular_tds.append(TensorDict(regular_items, batch_size=td.batch_size, device=td.device))

    # Concatenate the regular tensordicts
    output = TensorDict.cat(regular_tds, dim=0)

    # Concatenate and add nested tensors to the output
    for key in nested_tensor_keys:
        nested_tensors_to_concat = [td[key] for td in data]
        output[key] = concat_nested_tensors(nested_tensors_to_concat)

    return output


def chunk_tensordict(td: TensorDict, chunks: int) -> list[TensorDict]:
    """Split a TensorDict into equal-sized chunks with special nested tensor handling.

    Divides a TensorDict into the specified number of chunks along the batch
    dimension. Handles NestedTensors specially since TensorDict.chunk() doesn't
    support jagged tensors.

    Args:
        td: The TensorDict to split.
        chunks: Number of chunks to create. Must evenly divide len(td).

    Returns:
        List of TensorDicts, each containing a portion of the original data.

    Raises:
        AssertionError: If td is not a TensorDict or if its length is not
            evenly divisible by chunks.

    Note:
        PyTorch ``unbind(dim=0)`` on 3D+ jagged NestedTensors has a bug where
        ``split_with_sizes`` is applied to the wrong dimension of the internal
        ``_values`` tensor.  For example, mRoPE ``position_ids`` with per-sample
        shape ``(4, seq_len)`` becomes a 3D jagged NestedTensor
        ``[B, *(ragged=4), seq_len]``; ``_values`` is ``[B*4, seq_len]`` and
        ``unbind`` erroneously splits dimension 1 (``seq_len``) instead of
        dimension 0, causing::

            RuntimeError: split_with_sizes expects split_sizes to sum exactly
            to <seq_len>, but got split_sizes=[4, 4, ...]

        2D jagged NestedTensors (e.g. ``input_ids``, ``loss_mask``) are
        unaffected — ``unbind(dim=0)`` works correctly for them.

        The workaround: try ``unbind`` first (fast path for 2D); on failure,
        fall back to ``to_padded_tensor`` → ``chunk`` → reconstruct per-chunk
        NestedTensors using the original ragged lengths from ``offsets``.

        See https://github.com/pytorch/pytorch/issues/153238
    """
    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (
        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
    )
    chunk_size = len(td) // chunks
    nested_keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested}
    new_td = TensorDict(
        {k: v for k, v in td.items() if k not in nested_keys}, batch_size=td.batch_size, device=td.device
    )

    tds = new_td.chunk(chunks=chunks)
    for key in nested_keys:
        nt = td[key]
        try:
            tensors = nt.unbind(dim=0)
        except RuntimeError:
            padded = nt.to_padded_tensor(0)
            padded_chunks = padded.chunk(chunks, dim=0)
            offsets = nt.offsets()
            lengths = offsets.diff().tolist()
            for i, chunk_td in enumerate(tds):
                chunk_lengths = lengths[i * chunk_size : (i + 1) * chunk_size]
                chunk_tensors = [padded_chunks[i][j, :seq_len] for j, seq_len in enumerate(chunk_lengths)]
                chunk_td[key] = nested_tensor_from_tensor_list(
                    chunk_tensors, ragged_idx=getattr(nt, "_ragged_idx", nt.dim() - 1)
                )
            continue

        for i, chunk_td in enumerate(tds):
            chunk_td[key] = nested_tensor_from_tensor_list(
                list(tensors[i * chunk_size : (i + 1) * chunk_size]),
                ragged_idx=getattr(nt, "_ragged_idx", nt.dim() - 1),
            )

    return tds


def get_tensordict(tensor_dict: dict[str, torch.Tensor | list], non_tensor_dict: dict = None) -> TensorDict:
    """Create a TensorDict from tensors and non-tensor data.

    Automatically handles nested structures in lists by converting them to NonTensorStack.
    This enables support for:
    - Lists of lists: [[], [0.5, 0.8], [0.9]]
    - Lists of dicts: [{"acc": 1.0}, {"acc": 0.0}]
    - Lists of lists of dicts: [[{"content": "...", "role": "user"}]]

    Args:
        tensor_dict: Dictionary of tensors and lists to include in the TensorDict
        non_tensor_dict: Dictionary of metadata to store as NonTensorData

    Returns:
        TensorDict with proper handling of nested structures

    Example:
        >>> td = get_tensordict(
        ...     tensor_dict={
        ...         "obs": torch.randn(3, 4),
        ...         "turn_scores": [[], [0.5, 0.8], [0.9]]  # Nested list
        ...     },
        ...     non_tensor_dict={"experiment": "test"}
        ... )
    """
    tensor_dict = tensor_dict.copy()
    if non_tensor_dict is None:
        non_tensor_dict = {}

    batch_size = None

    for key, val in tensor_dict.items():
        if isinstance(val, torch.Tensor) and val.is_nested:
            assert val.is_contiguous(), "Nested tensors must be contiguous. Try setting layout=torch.jagged"
            assert val.layout == torch.jagged, "Nested tensors must be jagged."

        # Skip validation for NonTensorStack as it's already properly formatted
        if isinstance(val, NonTensorStack):
            if batch_size is None:
                batch_size = len(val)
            else:
                assert len(val) == batch_size, (
                    f"Batch size of NonTensorStack {key} is not consistent with other tensors. "
                    f"Expected {batch_size}, got {len(val)}"
                )
            continue

        if isinstance(val, list | np.ndarray):
            for v in val:
                assert not isinstance(v, torch.Tensor), (
                    "Passing a list makes the data NonTensorStack, "
                    "which doesn't support torch.Tensor. Please convert to numpy first"
                )
            # Convert to NonTensorStack to handle nested structures
            tensor_dict[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])

        assert isinstance(val, torch.Tensor | list | np.ndarray), (
            f"{key} -> {type(val)} isn't of 'torch.Tensor | list | np.ndarray' type"
        )

        if batch_size is None:
            batch_size = val.size(0) if isinstance(val, torch.Tensor) else len(val)
        else:
            val_batch_size = val.size(0) if isinstance(val, torch.Tensor) else len(val)
            assert val_batch_size == batch_size, (
                f"Batch size of tensor {key} is not consistent with other tensors. "
                f"Expected {batch_size}, got {val_batch_size}"
            )

    if batch_size is None:
        batch_size = []
    else:
        batch_size = [batch_size]

    for key, val in non_tensor_dict.items():
        assert key not in tensor_dict
        tensor_dict[key] = NonTensorData(val)

    return TensorDict(source=tensor_dict, batch_size=batch_size)


def contiguous(data: TensorDict) -> TensorDict:
    """Call contiguous on a tensor dict. The contiguous function of tensordict lib will make NonTensorStack.
    This function will always return a new tensordict

    Args:
        data: The input tensordict

    Returns:
        a tensordict that is contiguous

    """
    tensor_dict = {}
    non_tensor_dict = {}

    for key in data.keys():
        val = data.get(key)
        if isinstance(val, NonTensorData):
            non_tensor_dict[key] = val
        elif isinstance(val, NonTensorStack):
            tensor_dict[key] = val
        else:
            assert isinstance(val, torch.Tensor), f"Expect val to be a torch.Tensor. Got {type(val)}"
            tensor_dict[key] = val.contiguous()

    return get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def index_select_tensor_dict(batch: TensorDict, indices: torch.Tensor | list[int]) -> TensorDict:
    """Select rows from a TensorDict using indices.

    Creates a new TensorDict containing only the rows specified by indices.
    Handles regular tensors, nested tensors, NonTensorStack, and NonTensorData
    appropriately.

    Args:
        batch: The TensorDict to index into. Can be None.
        indices: 1D tensor or list of integers specifying which rows to select.

    Returns:
        A new TensorDict containing only the selected rows, or None if
        batch was None.

    Raises:
        AssertionError: If indices is not 1-dimensional.

    Note:
        - Regular tensors are indexed directly
        - Nested tensors are unbound, indexed, and rebound
        - NonTensorStack is indexed by batch dimension
        - NonTensorData (scalar metadata) is preserved unchanged
    """
    if isinstance(indices, list):
        indices = torch.tensor(indices)

    assert indices.dim() == 1, "indices must be a 1D tensor"

    data_dict = {}
    batch_size = indices.shape[0]

    if batch is not None:
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                data_dict[key] = tensor[indices]
            elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                tensor_lst = tensor.unbind()  # for performance
                selected_tensors = [tensor_lst[idx] for idx in indices]
                data_dict[key] = nested_tensor_from_tensor_list(
                    selected_tensors, ragged_idx=getattr(tensor, "_ragged_idx", tensor.dim() - 1)
                )
            else:
                # This handles NonTensorStack (indexable by batch dim) and NonTensorData (scalar metadata).
                if tensor.shape:
                    data_dict[key] = tensor[indices]
                else:
                    data_dict[key] = tensor
        selected_batch = TensorDict(source=data_dict, batch_size=batch_size)
    else:
        selected_batch = None

    return selected_batch


def get(tensordict: TensorDict, key: str, default=None) -> Any:
    """Get a value from a TensorDict with automatic unwrapping.

    Retrieves a value from the TensorDict and automatically converts it
    to a Python-native format:
    - Tensors are returned as-is
    - NonTensorStack is converted to a Python list
    - NonTensorData is unwrapped to its underlying value

    Args:
        tensordict: The TensorDict to retrieve from.
        key: The key to look up.
        default: Value to return if the key doesn't exist. Defaults to None.

    Returns:
        The value for the key in its native format, or default if not found.

    Example:
        >>> td = get_tensordict({"obs": torch.randn(3, 4), "labels": ["a", "b", "c"]})
        >>> get(td, "obs")  # Returns torch.Tensor
        >>> get(td, "labels")  # Returns ["a", "b", "c"] as a list
        >>> get(td, "missing", "default")  # Returns "default"
    """
    if key not in tensordict:
        return default

    output = tensordict.get(key)
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, NonTensorStack):
        return output.tolist()
    else:
        assert isinstance(output, NonTensorData)
        return output.data
