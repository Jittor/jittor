Shared benchmark model definitions (kept identical between frameworks):

mlp        : Linear(1024,4096) -> ReLU -> Linear(4096,4096) -> ReLU
             -> Linear(4096,4096) -> ReLU -> Linear(4096,1000)
             batch 256, float32
cnn        : 6 conv blocks (3->64->64->128->128->256->256, 3x3 pad 1),
             maxpool after each pair, then Linear(256*4*4, 512) -> ReLU -> Linear(512,1000)
             input 64x3x32x32, float32
transformer: 4 x pre-norm block, d_model 512, heads 8, ffn 2048, GELU,
             input 8x256 token ids, vocab 8192, tied nothing, float32

Training step: forward -> cross entropy -> backward -> SGD(momentum=0.9) step
Inference step: forward only, no grad
