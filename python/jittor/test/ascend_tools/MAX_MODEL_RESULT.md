# TODO#1: Max trainable model on 8x 910B3 (64GB each)

## Method
Jittor multi-card is data-parallel (HCCL): every card holds a full model
replica, so the model-size ceiling is bounded by ONE card's 64GB HBM; 8 cards
multiply throughput (~linear), not model capacity. Probed a GPT-2-style
decoder (vocab 50257, seqlen 512, batch 1, fp32, Adam) scaling hidden size on a
single 910B3 to the OOM boundary.

## Result (fp32, Adam, B=1, T=512)
| config (L/H/E)      | params | result |
|---------------------|--------|--------|
| 24 / 16 / 2048      | 1.4B   | OK     |
| 32 / 20 / 2560      | 2.8B   | OK     |  <- max stable
| 32 / 24 / 3072      | ~3.9B  | OOM    |
| 32 / 28 / 3584      | ~5B    | OOM    |
| 32 / 32 / 4096      | ~7B    | OOM    |

=> Largest model that TRAINS on a single 64GB 910B3 (and thus under 8-card
data-parallel, at 8x throughput): ~2.8B parameters.

## What bounds it
fp32 Adam keeps weight + grad + 2 optimizer moments = 16 bytes/param (~45GB at
2.8B) plus activations + the large vocab head (50257xE). The OOM first shows up
in the embedding/vocab-head getitem.

## Going bigger (notes)
- fp16/bf16 weights+grads would roughly double capacity, but this ACL build's
  mixed-precision Adam path is currently unstable (aborts), so the *reliable*
  number is the fp32 2.8B above.
- True >64GB models need tensor/pipeline (model) parallelism to use the
  aggregate 512GB; Jittor's HCCL path here is data-parallel only. That is a
  larger build-out (sharded layers + collective all-gather/reduce-scatter).
