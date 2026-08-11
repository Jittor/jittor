"""Legacy CUDA inference comparison retained for historical investigations.

New performance coverage belongs in the repository ASV suite. Importing this
module deliberately does not import either framework or initialize CUDA.
"""

from __future__ import print_function

import argparse
import time


DEFAULT_MODELS = (
    "squeezenet1_1",
    "alexnet",
    "resnet50",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "vgg11",
    "wide_resnet50_2",
    "wide_resnet101_2",
)


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_batch_sizes(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return values


def _to_cuda(value):
    return value.cuda()


def _run_model(np, jt, torch, jtmodels, tcmodels, model_name, batch_size, total):
    image_size = 300 if model_name == "inception_v3" else 224
    test_image = np.random.random((batch_size, 3, image_size, image_size)).astype("float32")
    torch_input = _to_cuda(torch.Tensor(test_image))
    jittor_input = jt.array(test_image).stop_grad()

    torch_model = _to_cuda(tcmodels.__dict__[model_name]())
    jittor_model = jtmodels.__dict__[model_name]()
    torch_model.eval()
    jittor_model.eval()
    jittor_model.load_parameters(torch_model.state_dict())

    iterations = max(2, total // batch_size)
    warmup = max(2, iterations // 8)
    print("{} {} {}".format("=" * 12, model_name, "=" * 12))

    for _index in range(warmup):
        jittor_result = jittor_model(jittor_input)
    jt.sync_all(True)
    started = time.time()
    for _index in range(iterations):
        jittor_result = jittor_model(jittor_input)
        jittor_result.sync()
    jt.sync_all(True)
    elapsed = time.time() - started
    print(
        "Jittor forward: {:.6f} s, batch={}, FPS={:.2f}".format(
            elapsed / iterations, batch_size, batch_size * iterations / elapsed
        )
    )

    for _index in range(warmup):
        torch_result = torch_model(torch_input)
    torch.cuda.synchronize()
    started = time.time()
    for _index in range(iterations):
        torch_result = torch_model(torch_input)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    print(
        "PyTorch forward: {:.6f} s, batch={}, FPS={:.2f}".format(
            elapsed / iterations, batch_size, batch_size * iterations / elapsed
        )
    )

    expected = torch_result.detach().cpu().numpy() + 1
    actual = jittor_result.numpy() + 1
    relative_error = np.abs(expected - actual) / np.abs(actual)
    difference = float(relative_error.mean())
    if difference >= 1e-3:
        raise AssertionError(
            "{} forward relative error is too large: {}".format(model_name, difference)
        )
    print("{} forward relative error: {:.8g}".format(model_name, difference))
    torch.cuda.empty_cache()
    jt.clean()
    jt.gc()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=(1, 2, 4, 8))
    parser.add_argument("--total-images", type=_positive_int, default=512)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args(argv)

    import numpy as np
    import jittor as jt
    import jittor.models as jtmodels
    import torch
    import torchvision.models as tcmodels

    if not jt.has_cuda:
        raise SystemExit("Jittor CUDA is required for this legacy benchmark")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is required for this legacy benchmark")

    missing = [
        name
        for name in args.models
        if name not in jtmodels.__dict__ or name not in tcmodels.__dict__
    ]
    if missing:
        raise SystemExit("unknown model(s): {}".format(", ".join(missing)))

    jt.flags.use_cuda = 1
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    jt.cudnn.set_algorithm_cache_size(10000)

    with torch.no_grad():
        for batch_size in args.batch_sizes:
            for model_name in args.models:
                _run_model(
                    np,
                    jt,
                    torch,
                    jtmodels,
                    tcmodels,
                    model_name,
                    batch_size,
                    args.total_images,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
