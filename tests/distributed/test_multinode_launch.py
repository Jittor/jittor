"""Real loopback agent/Store contracts, with GPU discovery explicitly stubbed.

These host tests exercise the public launcher and OS worker lifetime handling.
They prove no NCCL execution or actual multi-host connectivity.
"""

import json
from pathlib import Path
import sys

import pytest

from _helpers.child_process import run_python_child


ROOT = Path(__file__).resolve().parents[2]
DRIVER = r'''
import ctypes, json, os, pathlib, signal, socket, subprocess, sys, time
root, output, scenario = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0,str(root/'tests'))
from _helpers.child_process import PYTHON, child_env
output.mkdir(parents=True, exist_ok=True)
# Adopt and reap workers if their agent is deliberately killed.
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
bootstrap = output / 'launcher.py'
bootstrap.write_text("""import sys, types, runpy
jt=types.ModuleType('jittor'); jt.__path__=[]
backend=types.ModuleType('jittor.compile_extern')
backend._skip_nccl_p2p_without_peer_access=lambda:None
sys.modules.update({'jittor':jt,'jittor.compile_extern':backend})
sys.argv[0]=LAUNCH
runpy.run_path(LAUNCH,run_name='__main__')
""".replace('LAUNCH', repr(str(root / 'python/jittor/distributed/launch.py'))))
rank_script = output / 'rank.py'
rank_script.write_text("""import importlib.util, json, os, pathlib, sys, time
spec=importlib.util.spec_from_file_location('store', STORE)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
store,rank,world=next(module.rendezvous('env://',timeout=5))
store.set('fixture/rank/'+str(rank),str(rank))
store.wait(['fixture/rank/'+str(peer) for peer in range(world)])
data={key:os.environ[key] for key in ('RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE')}
data['pid']=os.getpid()
pathlib.Path(OUTPUT,'worker'+str(rank)+'.json').write_text(json.dumps(data))
if SCENARIO=='failure' and rank==world-1: sys.exit(7)
if SCENARIO!='success': time.sleep(60)
store.close()
""".replace('STORE',repr(str(root/'python/jittor/distributed/store.py')))
    .replace('OUTPUT',repr(str(output))).replace('SCENARIO',repr(scenario)))
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
procs=[]
nproc=2 if scenario=='success' else 1
try:
    for node in (0,1):
        cmd=[PYTHON,str(bootstrap),'--nnodes=2','--node-rank='+str(node),
             '--rdzv-id=host-fixture','--nproc-per-node='+str(nproc),'--backend=nccl',
             '--device-ids=fixture0,fixture1','--master-addr=127.0.0.1','--master-port='+str(port),
             '--timeout=2','--log-dir='+str(output/('agent'+str(node))),str(rank_script)]
        stream=(output/('agent%d.log'%node)).open('w')
        procs.append((subprocess.Popen(cmd, env=child_env(), stdout=stream, stderr=subprocess.STDOUT),stream))
    if scenario=='agent-death':
        deadline=time.monotonic()+10
        while len(list(output.glob('worker*.json')))<2:
            if time.monotonic()>deadline:raise AssertionError('workers did not start')
            time.sleep(.02)
        procs[1][0].kill()
    codes=[proc.wait(timeout=15) for proc,_ in procs]
    if scenario=='success':
        assert codes==[0,0],codes
        workers=[json.loads((output/('worker%d.json'%rank)).read_text()) for rank in range(4)]
        assert [x['RANK'] for x in workers]==['0','1','2','3']
        assert all(x['WORLD_SIZE']=='4' and x['LOCAL_WORLD_SIZE']=='2' for x in workers)
        assert [x['GROUP_RANK'] for x in workers]==['0','0','1','1']
    else:
        assert all(code!=0 for code in codes),codes
    deadline=time.monotonic()+3
    for file in output.glob('worker*.json'):
        pid=json.loads(file.read_text())['pid']
        while True:
            try:os.waitpid(pid,os.WNOHANG)
            except ChildProcessError:pass
            try:os.kill(pid,0)
            except ProcessLookupError:break
            if time.monotonic()>deadline:raise AssertionError('live orphan worker '+str(pid))
            time.sleep(.02)
    print(json.dumps({'scenario':scenario,'returncodes':codes}))
finally:
    for proc,stream in procs:
        if proc.poll() is None:proc.kill();proc.wait()
        stream.close()
'''


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux worker lifetime contract')
@pytest.mark.parametrize('scenario', ['success', 'failure', 'agent-death'])
def test_public_agents_use_global_ranks_and_propagate_failure(tmp_path, scenario):
    result = run_python_child(['-c', DRIVER, str(ROOT), str(tmp_path), scenario],
                              merge_stderr=True, timeout=45, crash_isolated=True)
    logs = '\n'.join(path.read_text() for path in tmp_path.glob('agent*.log'))
    assert result.returncode == 0, result.stdout + '\n' + logs
    assert json.loads(result.stdout.splitlines()[-1])['scenario'] == scenario
