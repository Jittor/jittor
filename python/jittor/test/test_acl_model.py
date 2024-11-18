import jittor as jt
from jittor import nn
import numpy as np
import time
import argparse
from typing import Callable, Dict, Optional
from dataclasses import dataclass

jt.flags.use_cuda = 1


@dataclass
class Config:
    model_name: str = "all"
    bs: int = 64
    repeat_num: int = 10
    is_train: bool = False


def get_parser() -> Config:
    parser = argparse.ArgumentParser(description="Test ACL model")
    parser.add_argument("--model_name",
                        type=str,
                        default="all",
                        help="Model name")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--repeat_num",
                        type=int,
                        default=10,
                        help="Repeat number")
    parser.add_argument("--is_train",
                        action="store_true",
                        help="Whether to train")
    args = parser.parse_args()
    return Config(**vars(args))


def measure_time(func: Callable) -> Callable:
    """时间测量装饰器"""

    def wrapper(*args, **kwargs):
        for _ in range(5):  # Warm-up
            result = func(*args, **kwargs)
            sync_result(result)

        total_time = 0
        for _ in range(args[0].config.repeat_num):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            sync_result(result)
            total_time += time.perf_counter() - start

        avg_time = total_time / args[0].config.repeat_num
        print(f"{func.__name__} executed in {avg_time * 1000:.6f} ms")
        return result

    return wrapper


def sync_result(result):
    """统一同步操作"""
    if isinstance(result, (list, tuple)):
        for item in result:
            item.sync()
    else:
        result.sync()


class TestACLModel:

    def __init__(self, config: Config):
        self.config = config
        self.test_tasks: Dict[str, Callable] = {
            "resnet18": self.resnet18,
            "linear": self.linear,
            "attention": self.attention,
            "ffn": self.ffn,
        }

    def test(self):
        if self.config.model_name == "all":
            for task in self.test_tasks.values():
                task()
        else:
            task = self.test_tasks.get(self.config.model_name)
            if task:
                task()
            else:
                print(f"Unknown model name: {self.config.model_name}")

    @measure_time
    def resnet18(self):
        from jittor.models import resnet18
        model = resnet18()
        x = jt.random([self.config.bs, 3, 224, 224])
        if self.config.is_train:
            self.train_step(model, x, "resnet_train")
        else:
            return model(x)

    @measure_time
    def linear(self):
        model = nn.Linear(1000, 1000)
        x = jt.random([self.config.bs, 1000])
        if self.config.is_train:
            self.train_step(model, x, "single_linear_train")
        else:
            return model(x)

    @measure_time
    def attention(self):
        from jittor.attention import MultiheadAttention
        model = MultiheadAttention(512, 8, self_attention=True)
        x = jt.random([self.config.bs, 1000, 512])
        if self.config.is_train:
            self.train_step(model, x, "attention_linear_train")
        else:
            return model(x)

    @measure_time
    def ffn(self):
        model = FeedForward(1024, 1024, multiple_of=256)
        x = jt.random([self.config.bs, 1024])
        if self.config.is_train:
            self.train_step(model, x, "ffn_linear_train")
        else:
            return model(x)

    def train_step(self, model, x, name):
        optimizer = jt.optim.SGD(model.parameters(), lr=0.0001)
        model.train()
        y = model(x)
        loss = y.mean()
        optimizer.step(loss)
        print(f"{name}: Loss = {loss.item()}")
        return loss


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def execute(self, x):
        return self.w2(jt.nn.silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__":
    config = get_parser()
    tester = TestACLModel(config)
    tester.test()
