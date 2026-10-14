"""Minimal native single-node DDP training example."""

import numpy as np

import jittor as jt
from jittor import nn
from jittor import distributed as dist
from jittor.dataset import Dataset
from jittor.nn.parallel import DistributedDataParallel


class SyntheticRegression(Dataset):
    def __init__(self):
        super().__init__(
            batch_size=32, shuffle=True, drop_last=True, num_workers=0, seed=17)
        self.set_attrs(total_len=1024)

    def __getitem__(self, index):
        x = np.asarray([index / 1024.0], dtype=np.float32)
        y = np.asarray([2.0 * x[0] - 0.5], dtype=np.float32)
        return x, y


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def execute(self, x):
        return self.linear(x)


def main():
    dist.init_process_group(backend="nccl")
    try:
        model = DistributedDataParallel(Regressor())
        optimizer = jt.optim.Adam(model.parameters(), lr=1e-2)
        dataset = SyntheticRegression()

        for epoch in range(5):
            dataset.set_epoch(epoch)
            losses = []
            for x, target in dataset:
                loss = ((model(x) - target) ** 2).mean()
                optimizer.backward(loss)
                optimizer.step()
                losses.append(float(loss.numpy().mean()))
            if dist.get_rank() == 0:
                print("epoch={} loss={:.6f}".format(
                    epoch, sum(losses) / max(1, len(losses))))

        if dist.get_rank() == 0:
            model.module.save("native-ddp-checkpoint.pkl")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
