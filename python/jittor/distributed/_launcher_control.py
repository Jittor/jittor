"""Agent control plane over the native TCPStore; no tensor-runtime import.

Only launch agents host servers. GPU workers may hold the Python GIL for a
collective, so neither their progress nor their shutdown owns this service.
Files containing NCCL heartbeats remain local to each node; agents exchange
node liveness and failures through the store instead of assuming shared disks.
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import time


def _store_module():
    # The public launcher also runs directly by path, before Jittor is loaded.
    # Load the same physical native Store owner, not a second implementation.
    spec = importlib.util.spec_from_file_location(
        '_jittor_launcher_store', Path(__file__).with_name('store.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgentControl:
    def __init__(self, args, backend, host, port, rootinfo):
        module = _store_module()
        prefix = module.launcher_store_prefix(args.rdzv_id)
        self.raw = module.TCPStore(host, port, is_master=args.node_rank == 0,
                                   timeout=args.timeout, wait_for_workers=False)
        self.store = module.PrefixStore(prefix, self.raw)
        self.node = args.node_rank
        self.nodes = args.nnodes
        self.nproc = args.nproc
        self.timeout = args.timeout
        self.rootinfo = rootinfo
        self.sequence = 0
        self.last_seen = {}
        self.local_beats = {}
        self.local_done = False
        self.active_ranks = set(range(self.node * self.nproc, (self.node + 1) * self.nproc))
        self.spec = {'nnodes': args.nnodes, 'nproc': args.nproc, 'backend': backend}
        try:
            self.interval = float(os.environ.get('JT_NCCL_WATCHDOG_INTERVAL_S', '5'))
            self.stale = float(os.environ.get('JT_NCCL_WATCHDOG_STALE_S', str(4 * self.interval)))
            if (not math.isfinite(self.interval) or not math.isfinite(self.stale)
                    or (self.interval > 0 and self.stale <= 0)):
                raise ValueError('multi-node NCCL watchdog intervals must be finite, with a positive stale budget')
            if self.store.add('nodes/{}/claims'.format(self.node), 1) != 1:
                raise ValueError('duplicate launcher node rank {}'.format(self.node))
            self.store.set('nodes/{}/spec'.format(self.node), json.dumps(self.spec, sort_keys=True))
            deadline = time.monotonic() + self.timeout
            keys = ['nodes/{}/spec'.format(node) for node in range(self.nodes)]
            while not self.store.check(keys):
                if time.monotonic() >= deadline:
                    raise TimeoutError('launcher timed out waiting for {} node agents'.format(self.nodes))
                time.sleep(.05)
            if any(json.loads(self.store.get(key)) != self.spec for key in keys):
                raise ValueError('launcher node agents disagree on backend/node/local-worker counts')
            # Thereafter control requests are short, even when a peer disappears.
            self.raw.set_timeout(min(5., self.timeout))
            self.tick()
        except (OSError, RuntimeError, ValueError, TimeoutError):
            self.raw.close()
            raise

    def fail(self, message):
        self.store.set('abort', json.dumps({'node_rank': self.node, 'error': str(message)}))

    def _watch_local_workers(self, now):
        if self.interval <= 0:
            return
        for rank in self.active_ranks:
            key = 'workers/jittor/nccl/world/initialized/{}'.format(rank)
            # A cold compiler or a pure host launcher fixture has no initialized
            # NCCL communicator and therefore owes no native heartbeat yet.
            if not self.store.check([key]):
                continue
            previous, changed = self.local_beats.setdefault(rank, (None, now))
            try:
                stamp = os.stat(self.rootinfo + '.hb' + str(rank)).st_mtime_ns
            except FileNotFoundError:
                stamp = None
            if stamp is not None and stamp != previous:
                self.local_beats[rank] = (stamp, now)
            elif now - changed > self.stale:
                raise RuntimeError('rank {} stopped updating its local NCCL heartbeat'.format(rank))

    def tick(self, active_ranks=None):
        if active_ranks is not None:
            self.active_ranks = set(active_ranks)
        now = time.monotonic()
        self.sequence += 1
        self.store.set('nodes/{}/status'.format(self.node), json.dumps(
            {'sequence': self.sequence, 'done': self.local_done}))
        if self.store.check(['abort']):
            raise RuntimeError('another launcher reported failure: ' + self.store.get('abort').decode('utf8'))
        for node in range(self.nodes):
            key = 'nodes/{}/status'.format(node)
            old, changed = self.last_seen.setdefault(node, (None, now))
            if self.store.check([key]):
                status = json.loads(self.store.get(key))
                if status['sequence'] != old:
                    self.last_seen[node] = (status['sequence'], now)
                    continue
            if now - changed > self.timeout:
                raise RuntimeError('launcher node {} stopped updating its heartbeat'.format(node))
        if not self.local_done:
            self._watch_local_workers(now)

    def finish(self):
        self.local_done = True
        keys = ['nodes/{}/status'.format(node) for node in range(self.nodes)]
        while True:
            self.tick()
            if (self.store.check(keys)
                    and all(json.loads(self.store.get(key))['done'] for key in keys)):
                break
            time.sleep(.05)
        # Reply-before-marker arrival prevents the server owner from closing
        # before another agent's final reply has been flushed.
        self.store.arrive('exit_ack/{}'.format(self.node))
        if self.node == 0:
            self.store.wait(['exit_ack/{}'.format(node) for node in range(self.nodes)], self.timeout)

    def close(self):
        self.raw.close()
