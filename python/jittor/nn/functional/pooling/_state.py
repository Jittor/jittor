"""One live legacy pooling switch shared by all public spellings."""

from types import ModuleType
import sys

pool_use_code_op = True

class PoolingStateView(ModuleType):
    @property
    def pool_use_code_op(self):
        return getattr(sys.modules[__name__], "pool_use_code_op")

    @pool_use_code_op.setter
    def pool_use_code_op(self, value):
        global pool_use_code_op
        pool_use_code_op = value

    @pool_use_code_op.deleter
    def pool_use_code_op(self):
        # Temporary attribute overrides restore the original via this same
        # property; deleting the owner entry preserves normal hasattr behavior.
        global pool_use_code_op
        del pool_use_code_op
