"""Honest unsupported builder implementing the DeepSpeed builder protocol."""

from deepspeed.ops.op_builder.builder import OpBuilder


class UnsupportedBuilder(OpBuilder):
    NAME = "deepspeed_not_implemented"
    BUILD_VAR = "DS_BUILD_NOT_IMPLEMENTED"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def is_compatible(self, verbose=False):
        return False

    def load(self, verbose=True):
        raise NotImplementedError(
            "This eager-only provider cannot load PyTorch ABI extensions; "
            "pass a supported public torch optimizer explicitly.")

    def jit_load(self, verbose=True):
        return self.load(verbose=verbose)

    def absolute_name(self):
        return "jittor_adapters.deepspeed.unsupported_extension"

    def sources(self):
        return []
