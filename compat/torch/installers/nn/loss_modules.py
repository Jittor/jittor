"""Stable loss and pixel-rearrangement layer implementations."""
import jittor as jt
from jittor import Module
from ...context import get_install_context


class _FunctionalLoss(Module):
    defaults = ()
    argument_order = ()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._kw = dict(self.defaults)
        self._kw.update(kwargs)
        for name, value in zip(self.argument_order, args):
            self._kw[name] = value

    def execute(self, *inputs):
        functions = get_install_context(jt).state["nn_functional_native"]["loss_functions"]
        return functions[self.functional_name](*inputs, **self._kw)


class HuberLoss(_FunctionalLoss):
    functional_name = "huber_loss"
    defaults = (("reduction", "mean"), ("delta", 1.0))
    argument_order = ("reduction", "delta")


class SmoothL1Loss(_FunctionalLoss):
    functional_name = "smooth_l1_loss"
    defaults = (("reduction", "mean"),)
    argument_order = ("reduction",)


class MarginRankingLoss(_FunctionalLoss):
    functional_name = "margin_ranking_loss"
    defaults = (("margin", 0.0), ("reduction", "mean"))
    argument_order = ("margin", "reduction")


class CosineEmbeddingLoss(MarginRankingLoss):
    functional_name = "cosine_embedding_loss"


class GaussianNLLLoss(_FunctionalLoss):
    functional_name = "gaussian_nll_loss"
    defaults = (("full", False), ("eps", 1e-6), ("reduction", "mean"))
    argument_order = ("full", "eps", "reduction")


class NLLLoss(_FunctionalLoss):
    functional_name = "nll_loss"
    defaults = (("reduction", "mean"),)
    argument_order = ("weight", "size_average", "ignore_index")


class PixelShuffle(Module):
    def __init__(self, factor):
        super().__init__()
        self._f = factor

    def execute(self, value):
        return get_install_context(jt).target_namespace.nn.functional.pixel_shuffle(value, self._f)


class PixelUnshuffle(PixelShuffle):
    def execute(self, value):
        return get_install_context(jt).target_namespace.nn.functional.pixel_unshuffle(value, self._f)


LOSS_CLASSES = (HuberLoss, SmoothL1Loss, MarginRankingLoss, CosineEmbeddingLoss,
                GaussianNLLLoss, NLLLoss)
