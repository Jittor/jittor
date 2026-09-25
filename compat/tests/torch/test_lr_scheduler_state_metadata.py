"""Scheduler metadata only: no tensor, CPU model, or device computation."""
import copy
import functools
import pytest
import torch


class MetadataOptimizer(torch.optim.Optimizer):
    def __init__(self, groups=None):
        self.param_groups = groups if groups is not None else [{"lr": 0.125}]
        self.lr = self.param_groups[0]["lr"]
    def step(self):
        pass


class Factor:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, epoch):
        return self.factor


@pytest.mark.parametrize("name", ["LambdaLR", "MultiplicativeLR"])
def test_callable_checkpoint_preserves_receiving_functions(name):
    cls = getattr(torch.optim.lr_scheduler, name)
    fn = lambda epoch: 0.5
    fn.unsaved = "ordinary functions do not serialize attributes"
    obj = Factor(0.25)
    opt = MetadataOptimizer([{"lr": 0.125}, {"lr": 0.25}])
    scheduler = cls(opt, [fn, obj])
    opt.step()
    scheduler.step()
    saved = copy.deepcopy(scheduler.state_dict())
    assert set(saved) == {"base_lrs", "last_epoch", "_step_count", "_is_initial",
                          "_get_lr_called_within_step", "_last_lr", "lr_lambdas"}
    assert saved["lr_lambdas"] == [None, {"factor": 0.25}]
    assert saved["_is_initial"] is False
    assert saved["_get_lr_called_within_step"] is False
    target_fn = lambda epoch: 0.5
    target_obj = Factor(0.75)
    target_opt = MetadataOptimizer([{"lr": 0.125}, {"lr": 0.25}])
    target = cls(target_opt, [target_fn, target_obj])
    target_opt.param_groups = copy.deepcopy(opt.param_groups)
    before = copy.deepcopy(saved)
    target.load_state_dict(saved)
    assert saved == before
    assert target.lr_lambdas[0] is target_fn
    assert target.lr_lambdas[1] is target_obj
    assert target_obj.factor == 0.25
    opt.step(); scheduler.step()
    target_opt.step(); target.step()
    assert target.state_dict() == scheduler.state_dict()
    assert target_opt.param_groups == opt.param_groups


@pytest.mark.parametrize("name", ["LambdaLR", "MultiplicativeLR"])
def test_partial_and_group_validation(name):
    cls = getattr(torch.optim.lr_scheduler, name)
    fn = functools.partial(pow, 0.5)
    fn.label = "saved callable metadata"
    scheduler = cls(MetadataOptimizer(), fn)
    assert scheduler.state_dict()["lr_lambdas"] == [{"label": fn.label}]
    with pytest.raises(ValueError):
        cls(MetadataOptimizer(), [fn, fn])
    with pytest.raises(KeyError, match="initial_lr"):
        cls(MetadataOptimizer(), fn, last_epoch=2)


def test_initial_lr_resume_and_live_context_flags():
    events = []
    class Observed(torch.optim.lr_scheduler.LambdaLR):
        def get_lr(self):
            events.append((self._is_initial, self._get_lr_called_within_step,
                           self.last_epoch))
            return super().get_lr()
    opt = MetadataOptimizer([{"lr": 0.03125, "initial_lr": 0.125}])
    scheduler = Observed(opt, lambda epoch: 0.5 ** epoch, last_epoch=2)
    assert events == [(True, True, 3)]
    assert scheduler.base_lrs == [0.125]
    assert opt.param_groups[0]["lr"] == 0.015625
    opt.step(); scheduler.step()
    assert events[-1] == (False, True, 4)
    assert not scheduler._is_initial and not scheduler._get_lr_called_within_step


def test_context_flags_reset_when_get_lr_raises():
    objects = []
    class Broken(torch.optim.lr_scheduler.LambdaLR):
        def get_lr(self):
            objects.append(self)
            assert self._is_initial and self._get_lr_called_within_step
            raise ValueError("scheduler failure")
    with pytest.raises(ValueError, match="scheduler failure"):
        Broken(MetadataOptimizer(), lambda epoch: 1.0)
    assert objects[0]._is_initial is False
    assert objects[0]._get_lr_called_within_step is False


def test_multiplicative_resume_initial_call_and_external_lr():
    opt = MetadataOptimizer([{"lr": 0.03125, "initial_lr": 0.125}])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: 0.5,
                                                         last_epoch=2)
    assert scheduler.base_lrs == [0.125]
    assert scheduler.get_last_lr() == [0.03125]
    opt.param_groups[0]["lr"] = 0.0625
    opt.step(); scheduler.step()
    assert scheduler.get_last_lr() == [0.03125]
