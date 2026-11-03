# MiniMax-H3 on Jittor, through the torch compat layer

The serving integration now lives in its own repository:
**[JittorEcosystem/jittor-minimax-h3](https://github.com/JittorEcosystem/jittor-minimax-h3)**,
pinned to vLLM-Omni `v0.29.0rc1-58-g446c2b54` (commit `446c2b547`), vLLM
`0.29.0` (commit `98dff2a81`) and the `MiniMaxAI/MiniMax-H3` weights.

Text-to-video-and-audio serving with **unmodified vLLM-Omni** and **unmodified
MiniMax-H3 weights**: Jittor's `compat` layer stands in for PyTorch, and
vLLM-Omni is imported from its own checkout as-is. Jittor must be new enough to
contain the cross-thread compute-stream handoff
(`backend_compute_stream_acquire`); without it the VAE decode returns noise
while the server still answers 200.

What moved to that repository, and is maintained there rather than here:

- the `vllm-omni serve` flag set and the environment it needs
  (`jittor_minimax_h3.serve_argv` / `build_environment`);
- the create/poll/download request protocol, as runnable code;
- the `ref2va` reference workflow and the curl/`--form-string` trap;
- the recorded 8x H100-96GB measurements and the line between what was executed
  and what was not;
- the flash-attention bridge it depends on, which is
  [jittor-flash-attn](https://github.com/JittorEcosystem/jittor-flash-attn) and
  compiles its kernels on first use.

Start with that repository's README; this page is a pointer.
