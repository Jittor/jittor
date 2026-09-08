"""Failure-aware prerequisites and fixture lifetimes without native bootstrap."""
import builtins
import ast
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _helpers import capability
from _helpers.runtime_policy import fixture_stack, preserve_policy
from _helpers.introspection import liveness_snapshot


def record(state):
    return SimpleNamespace(name="probe",kind="library",state=SimpleNamespace(value=state),
                           reason="controlled "+state, enabled=state=="available",
                           failed=state=="failed",unprobed=state=="unprobed")


def test_helpers_import_without_bootstrapping_jittor(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name.split('.')[0] not in ('jittor','jittor_core'),name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    importlib.reload(capability)


@pytest.mark.parametrize('state',['failed','unprobed'])
def test_library_prerequisite_never_turns_a_failure_or_unprobed_into_skip(state):
    current=[record('unprobed')]
    calls=[]
    def load(name,load):
        assert load
        calls.append(name);current[0]=record(state)
        return current[0]
    backend=SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(library=lambda name:current[0])),
                            capability=SimpleNamespace(library=load))
    with pytest.raises(AssertionError):
        capability.library_enabled('probe',backend=backend)
    assert calls==['probe']


def test_library_decorator_defers_load_and_preserves_test_identity():
    calls=[];current=[record('unprobed')]
    def load(name,load):
        calls.append('load');current[0]=record('available');return current[0]
    backend=SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(library=lambda name:current[0])),
                            capability=SimpleNamespace(library=load))
    @capability.library_required('probe',backend=backend)
    def original_test(value):
        calls.append('run');return value
    assert calls==[]
    assert original_test.__name__=='original_test'
    assert original_test(7)==7
    assert calls==['load','run']


def test_library_class_prerequisite_preserves_the_actual_subclass():
    seen=[]
    backend=SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(
        library=lambda name:record('available'))))
    @capability.library_required('probe',backend=backend)
    class Parent(unittest.TestCase):
        @classmethod
        def setUpClass(cls):seen.append(cls)
    class Child(Parent):pass
    assert seen==[]
    Child.setUpClass()
    assert seen==[Child]


def test_failed_device_query_cannot_be_used_as_zero_devices():
    backend=SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(
        devices=lambda name:SimpleNamespace(capability=record('failed'),count=None))))
    with pytest.raises(AssertionError,match='broken build'):
        capability.device_count('cuda',backend=backend)


def test_fixture_policy_restores_after_a_failed_test():
    values={'mode':0};events=[]
    @contextmanager
    def scope(value):
        before=values['mode'];values['mode']=value
        try:yield
        finally:values['mode']=before;events.append('restore')
    class Example(unittest.TestCase):
        def setUp(self):
            fixture_stack(self).enter_context(scope(1))
        def runTest(self):
            assert values['mode']==1
            self.fail('deliberate assertion failure')
    result=unittest.TestResult();Example().run(result)
    assert len(result.failures)==1
    assert values['mode']==0 and events==['restore']


def test_python37_class_scope_restores_after_original_teardown():
    events=[]
    @contextmanager
    def scope():
        events.append('enter')
        try:yield
        finally:events.append('restore')
    class Example:
        @classmethod
        def tearDownClass(cls):events.append('teardown')
    fixture_stack(Example,class_scope=True).enter_context(scope())
    Example.tearDownClass()
    assert events==['enter','teardown','restore']


def test_teardown_scope_cannot_restore_a_dirty_test_value_after_cleanup():
    values=SimpleNamespace(mode=0)
    @contextmanager
    def scope(**changes):
        previous=values.mode;values.mode=changes['mode']
        try:yield
        finally:values.mode=previous
    backend=SimpleNamespace(runtime=SimpleNamespace(context=values,scope=scope))
    @preserve_policy(backend,'mode')
    class Example(unittest.TestCase):
        def setUp(self):self.saved=values.mode
        def runTest(self):values.mode=7
        def tearDown(self):
            with scope(mode=self.saved):
                assert values.mode==0
    result=unittest.TestResult();Example().run(result)
    assert result.wasSuccessful()
    assert values.mode==0


def test_generic_accelerator_sweep_keeps_acl_without_probing_cuda():
    calls=[]
    def query(name):
        calls.append(name)
        assert name=='acl'
        return record('available')
    backend=SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(
        registered_backends=lambda:('cpu','acl'), backend=query)))
    assert capability.any_accelerator_enabled(backend=backend)
    assert calls==['acl']


def test_liveness_projection_does_not_compare_execution_traffic():
    counters=SimpleNamespace(held_vars=2,live_vars=3,live_ops=1,exec_calls=4)
    backend=SimpleNamespace(introspection=SimpleNamespace(counters=counters))
    before=liveness_snapshot(backend)
    counters.exec_calls=99
    assert liveness_snapshot(backend)==before
    counters.live_vars=7
    assert liveness_snapshot(backend)['lived_vars']==7
    assert before['lived_vars']==3


def test_notebook_policy_lives_across_cells_and_restores_at_process_exit(monkeypatch):
    from types import ModuleType
    path=Path(__file__).resolve().parents[1]/'integration/test_notebooks.py'
    node=next(n for n in ast.parse(path.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='_offline_guard')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    flags=SimpleNamespace(use_cuda=1,use_parallel_op_compiler=0)
    @contextmanager
    def scope(**changes):
        before=flags.use_cuda;flags.use_cuda=changes['use_cuda']
        try:yield
        finally:flags.use_cuda=before
    backend=ModuleType('jittor')
    backend.runtime=SimpleNamespace(scope=scope)
    backend.introspection=SimpleNamespace(policy=SimpleNamespace(runtime=flags))
    socket=ModuleType('socket')
    socket.socket=SimpleNamespace(connect=lambda *args:None)
    socket.create_connection=lambda *args:None
    exits=[]
    atexit=ModuleType('atexit');atexit.register=exits.append
    for name,module in [('jittor',backend),('socket',socket),('atexit',atexit)]:
        monkeypatch.setitem(sys.modules,name,module)
    cells={}
    exec(namespace['_offline_guard'](),cells)
    exec('assert jt.introspection.policy.runtime.use_cuda == 0',cells)
    assert flags.use_cuda==0 and len(exits)==1
    exits.pop()()
    assert flags.use_cuda==1
