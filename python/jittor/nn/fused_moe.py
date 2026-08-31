"""Mixture-of-experts routing and expert compute.

A router scores every expert for every token, the top few win, and each winning
expert applies a SwiGLU feed-forward whose output is summed back with the router
weight. The two shapes that arise want different code: a single decoded token
touches exactly ``top_k`` experts and is best served by gathering just those
weights, while a prefill batch spreads across many experts and is best served by
visiting each active expert once with the tokens routed to it.
"""

import jittor as jt


def _router_weights(router_logits, top_k, renormalize, scoring):
    """Top-k expert ids and their combine weights, per token."""
    logits = router_logits.float32()
    if scoring == "sigmoid":
        scores = logits.sigmoid()
    else:
        scores = jt.nn.softmax(logits, dim=-1)
    weights, ids = jt.topk(scores, top_k, dim=-1)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdims=True) + 1e-20)
    return weights, ids


def fused_moe(x, router_logits, w13, w2, top_k, renormalize=True,
              scoring="softmax"):
    """Route ``x`` to its top experts and combine their SwiGLU outputs.

    ``x`` is ``[tokens, hidden]``, ``router_logits`` is ``[tokens, experts]``,
    ``w13`` is ``[experts, 2 * intermediate, hidden]`` holding the gate and up
    projections stacked, and ``w2`` is ``[experts, hidden, intermediate]``.
    ``scoring`` is ``"softmax"`` or ``"sigmoid"``.

    Computed in float32 and cast back to ``x``'s dtype.
    """
    hidden = int(x.shape[1])
    intermediate = int(w13.shape[1]) // 2
    tokens = int(x.shape[0])
    weights, ids = _router_weights(router_logits, top_k, renormalize, scoring)
    inputs = x.float32()

    if tokens == 1:
        # One token: gather only the experts it chose, on device. Visiting all
        # experts would do E gemms of work for k experts' worth of result, and
        # reading the ids on the host to pick them would stall the queue.
        chosen = ids.reshape(-1)
        gate_up = w13[chosen].float32()
        down = w2[chosen].float32()
        repeated = inputs.reshape(1, hidden).broadcast(
            [int(chosen.shape[0]), hidden])
        hidden_states = jt.bmm(
            repeated.unsqueeze(1), gate_up.transpose(1, 2))
        activated = (jt.nn.silu(hidden_states[:, :, :intermediate])
                     * hidden_states[:, :, intermediate:])
        expert_out = jt.bmm(activated, down.transpose(1, 2)).reshape(-1, hidden)
        combined = (weights.reshape(-1, 1) * expert_out).sum(0, keepdims=True)
        return combined.cast(x.dtype)

    # Many tokens: each active expert is visited once, with the rows routed to
    # it. Which experts are active has to be known on the host to slice them,
    # which costs one sync for the whole call rather than one per expert.
    import numpy as np

    out = jt.zeros((tokens, hidden), dtype=inputs.dtype)
    routed = np.asarray(ids.numpy())
    for expert in np.unique(routed.reshape(-1)):
        expert = int(expert)
        rows = np.where((routed == expert).any(axis=1))[0]
        if rows.size == 0:
            continue
        index = jt.array(rows.astype(np.int32))
        taken = inputs[index]
        # A token may pick the same expert in more than one slot; sum those.
        share = (weights[index] * (ids[index] == expert).float32()).sum(dim=-1)
        hidden_states = taken @ w13[expert].float32().transpose(0, 1)
        activated = (jt.nn.silu(hidden_states[:, :intermediate])
                     * hidden_states[:, intermediate:])
        expert_out = activated @ w2[expert].float32().transpose(0, 1)
        out[index] = out[index] + share.unsqueeze(-1) * expert_out
    return out.cast(x.dtype)


__all__ = ["fused_moe"]
