#!/usr/bin/env python3
import argparse
import sys

import jittor as torch


def assert_jittor_alias():
    is_jittor = sys.modules.get("torch") is torch and torch.__name__ == "jittor"
    print(f"TORCH_IS_JITTOR={is_jittor}")
    print(f"JITTOR_FILE={torch.__file__}")
    if not is_jittor:
        raise RuntimeError("TorchQuantum did not resolve torch imports to Jittor")


assert_jittor_alias()

import torchquantum as tq
import torchquantum.functional as tqf


def run_basic_usage(device):
    qdev = tq.QuantumDevice(
        n_wires=2, bsz=5, device=device, record_op=True
    )

    qdev.h(wires=0)
    qdev.cnot(wires=[0, 1])

    tqf.h(qdev, wires=1)
    tqf.x(qdev, wires=1)

    op = tq.RX(has_params=True, trainable=True, init_params=0.5)
    op(qdev, wires=0)

    print(qdev)

    from torchquantum.plugin import op_history2qasm

    qasm = op_history2qasm(qdev.n_wires, qdev.op_history)
    print(qasm)
    if "rx(0.5) q[0]" not in qasm:
        raise AssertionError("QASM output is missing the RX gate")

    samples = tq.measure(qdev, n_shots=1024)
    print(samples)

    from torchquantum.measurement import expval_joint_sampling

    expval_sampling = expval_joint_sampling(qdev, "ZX", n_shots=1024)
    print(expval_sampling)

    from torchquantum.measurement import expval_joint_analytical

    expval = expval_joint_analytical(qdev, "ZX")
    print(expval)
    expected = 0.87758256
    actual = float(expval[0].item())
    if abs(actual - expected) > 1e-5:
        raise AssertionError(
            f"analytical expectation mismatch: {actual} vs {expected}"
        )

    expval[0].backward()
    print(op.params.grad)
    if op.params.grad is None:
        raise AssertionError("RX parameter gradient is None")
    expected_grad = -0.47942554
    actual_grad = float(op.params.grad.reshape(-1)[0].item())
    if abs(actual_grad - expected_grad) > 1e-5:
        raise AssertionError(
            f"RX parameter gradient mismatch: {actual_grad} vs {expected_grad}"
        )

    ops = [
        {"name": "hadamard", "wires": 0},
        {"name": "cnot", "wires": [0, 1]},
        {
            "name": "rx",
            "wires": 0,
            "params": 0.5,
            "trainable": True,
        },
        {
            "name": "u3",
            "wires": 0,
            "params": [0.1, 0.2, 0.3],
            "trainable": True,
        },
        {"name": "h", "wires": 1, "inverse": True},
    ]

    qmodule = tq.QuantumModule.from_op_history(ops)
    qmodule(qdev)
    torch.sync_all(True)
    print(f"BASIC_DEVICE={device}")
    print("BASIC_USAGE_OK")


class QFCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.encoder_gates = (
            [tqf.rx] * 4
            + [tqf.ry] * 4
            + [tqf.rz] * 4
            + [tqf.rx] * 4
        )
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    def forward(self, x):
        bsz = x.shape[0]
        x = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)

        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device
        )

        for k, gate in enumerate(self.encoder_gates):
            gate(qdev, wires=k % self.n_wires, params=x[:, k])

        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])

        qdev.h(wires=3)
        qdev.sx(wires=2)
        qdev.cnot(wires=[3, 0])
        qdev.qubitunitary(
            wires=[1, 2],
            params=[
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1j],
                [0, 0, -1j, 0],
            ],
        )

        x = self.measure(qdev).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


def run_usage(device):
    model = QFCModel()
    x = torch.zeros((2, 1, 28, 28), device=device)
    output = model(x)
    torch.sync_all(True)
    print(output)
    if tuple(output.shape) != (2, 2):
        raise AssertionError(f"unexpected Usage output shape: {output.shape}")
    if not bool(torch.isfinite(output).all().item()):
        raise AssertionError("Usage output contains non-finite values")
    print(f"USAGE_DEVICE={device}")
    print("USAGE_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("basic", "usage"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.case == "basic":
        run_basic_usage(args.device)
    else:
        run_usage(args.device)


if __name__ == "__main__":
    main()
