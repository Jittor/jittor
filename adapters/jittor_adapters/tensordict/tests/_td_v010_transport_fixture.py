# Copyright (c) Meta Platforms, Inc. and affiliates.
# Exact TensorDict 0.10.0 helper; MIT-licensed upstream source.
def _rebuild_tensordict_files_consolidated(
    metadata,
    storage,
):
    def from_metadata(metadata=metadata, prefix=None):
        consolidated = {"storage": storage, "metadata": metadata}
        metadata = dict(metadata)
        non_tensor = metadata.pop("non_tensors")
        leaves = metadata.pop("leaves")
        cls = metadata.pop("cls")
        cls_metadata = dict(metadata.pop("cls_metadata"))
        is_locked = cls_metadata.pop("is_locked", False)
        # size can be there to tell what the size of the file is
        _ = metadata.pop("size", None)

        d = {
            key: NonTensorData(
                data,
                batch_size=batch_size,
                device=torch.device(device) if device is not None else None,
            )
            for (key, (data, batch_size, device)) in non_tensor.items()
        }
        for key, (dtype, local_shape, start, stop, pad) in leaves.items():
            dtype = _STR_DTYPE_TO_DTYPE[dtype]
            # device = torch.device(device)
            local_shape = torch.Size(local_shape)
            value = storage[start:stop].view(dtype)
            if pad:
                value = value[: local_shape.numel()]
            value = value.view(local_shape)
            if key.startswith("<NJT>"):
                raise RuntimeError
            elif key.startswith("<NJT_VALUES>"):
                nested_values = value
                nested_lengths = None
                continue
            elif key.startswith("<NJT_LENGTHS>"):
                nested_lengths = value
                continue
            elif key.startswith("<NJT_OFFSETS>"):
                from torch.nested._internal.nested_tensor import NestedTensor

                offsets = value
                value = NestedTensor(
                    nested_values, offsets=offsets, lengths=nested_lengths
                )
                key = key.replace("<NJT_OFFSETS>", "")
            d[key] = value
        for k, v in metadata.items():
            # Each remaining key is a tuple pointing to a sub-tensordict
            d[k] = from_metadata(
                v, prefix=prefix + (k,) if prefix is not None else (k,)
            )
        if isinstance(cls, str):
            cls = CLS_MAP[cls]
        result = cls._from_dict_validated(d, **cls_metadata)
        if is_locked:
            result = result.lock_()
        if _is_tensorclass(cls):
            result._tensordict._consolidated = consolidated
        else:
            result._consolidated = consolidated
        return result

    return from_metadata()
