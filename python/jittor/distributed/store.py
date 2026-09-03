"""Cross-process stores and rendezvous URL parsing for distributed jobs."""

import base64
import datetime
import json
import os
import socket
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlparse


_DEFAULT_TIMEOUT_SECONDS = 300.0


def _timeout_seconds(timeout):
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, datetime.timedelta):
        timeout = timeout.total_seconds()
    value = float(timeout)
    if value <= 0:
        raise ValueError("store timeout must be positive")
    return value


def _deadline(timeout):
    return time.monotonic() + _timeout_seconds(timeout)


def _remaining(deadline):
    return max(0.0, deadline - time.monotonic())


def _key_text(key):
    if isinstance(key, bytes):
        return key.decode("utf8")
    return str(key)


def _value_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf8")
    return str(value).encode("ascii")


class Store:
    """Thread-safe single-process Store base implementation."""

    def __init__(self, timeout=None):
        self._timeout = _timeout_seconds(timeout)
        self._data = {}
        self._condition = threading.Condition()

    def set_timeout(self, timeout):
        self._timeout = _timeout_seconds(timeout)

    def set(self, key, value):
        with self._condition:
            self._data[_key_text(key)] = _value_bytes(value)
            self._condition.notify_all()

    def get(self, key, timeout=None):
        name = _key_text(key)
        deadline = _deadline(self._timeout if timeout is None else timeout)
        with self._condition:
            while name not in self._data:
                remaining = _remaining(deadline)
                if not remaining:
                    raise TimeoutError("timed out waiting for store key {!r}".format(name))
                self._condition.wait(remaining)
            return self._data[name]

    def add(self, key, num):
        name = _key_text(key)
        with self._condition:
            value = int(self._data.get(name, b"0")) + int(num)
            self._data[name] = str(value).encode("ascii")
            self._condition.notify_all()
            return value

    def wait(self, keys, timeout=None):
        names = [_key_text(key) for key in keys]
        deadline = _deadline(self._timeout if timeout is None else timeout)
        with self._condition:
            while not all(name in self._data for name in names):
                remaining = _remaining(deadline)
                if not remaining:
                    missing = [name for name in names if name not in self._data]
                    raise TimeoutError(
                        "timed out waiting for store keys {}".format(missing)
                    )
                self._condition.wait(remaining)

    def check(self, keys):
        names = [_key_text(key) for key in keys]
        with self._condition:
            return all(name in self._data for name in names)

    def delete_key(self, key):
        with self._condition:
            return self._data.pop(_key_text(key), None) is not None

    def num_keys(self):
        with self._condition:
            return len(self._data)


class _TCPStoreServer:
    def __init__(self, host, port, timeout):
        self.store = Store(timeout)
        self.timeout = _timeout_seconds(timeout)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, int(port)))
        self.socket.listen()
        self._workers = 0
        self._workers_condition = threading.Condition()
        self._closed = False
        self._accept_thread = threading.Thread(target=self._accept_loop)
        self._accept_thread.daemon = True
        self._accept_thread.start()

    def _accept_loop(self):
        while not self._closed:
            try:
                connection, _address = self.socket.accept()
            except OSError:
                return
            thread = threading.Thread(
                target=self._serve_connection, args=(connection,)
            )
            thread.daemon = True
            thread.start()

    def _serve_connection(self, connection):
        stream = connection.makefile("rwb")
        try:
            while True:
                line = stream.readline()
                if not line:
                    return
                request = json.loads(line.decode("utf8"))
                try:
                    result = self._dispatch(request)
                    response = {"ok": True, "result": result}
                except (KeyError, TypeError, ValueError, TimeoutError) as error:
                    response = {"ok": False, "error": str(error)}
                stream.write((json.dumps(response) + "\n").encode("utf8"))
                stream.flush()
        finally:
            stream.close()
            connection.close()

    def _dispatch(self, request):
        operation = request["op"]
        if operation == "hello":
            with self._workers_condition:
                self._workers += 1
                self._workers_condition.notify_all()
            return self._workers
        if operation == "wait_workers":
            count = int(request["count"])
            deadline = _deadline(request.get("timeout", self.timeout))
            with self._workers_condition:
                while self._workers < count:
                    remaining = _remaining(deadline)
                    if not remaining:
                        raise TimeoutError(
                            "timed out waiting for {} TCPStore workers; got {}"
                            .format(count, self._workers)
                        )
                    self._workers_condition.wait(remaining)
            return self._workers
        if operation == "set":
            self.store.set(
                request["key"], base64.b64decode(request["value"])
            )
            return None
        if operation == "get":
            value = self.store.get(request["key"], request.get("timeout"))
            return base64.b64encode(value).decode("ascii")
        if operation == "add":
            return self.store.add(request["key"], request["value"])
        if operation == "wait":
            self.store.wait(request["keys"], request.get("timeout"))
            return None
        if operation == "check":
            return self.store.check(request["keys"])
        if operation == "delete":
            return self.store.delete_key(request["key"])
        if operation == "num_keys":
            return self.store.num_keys()
        raise ValueError("unsupported TCPStore operation {!r}".format(operation))

    def close(self):
        self._closed = True
        self.socket.close()


class _TCPStoreClient:
    def __init__(self, host, port, timeout):
        self.timeout = _timeout_seconds(timeout)
        deadline = _deadline(self.timeout)
        last_error = None
        while True:
            try:
                self.socket = socket.create_connection(
                    (host, int(port)), timeout=min(0.25, _remaining(deadline))
                )
                break
            except OSError as error:
                last_error = error
                if not _remaining(deadline):
                    raise TimeoutError(
                        "timed out connecting to TCPStore at {}:{}: {}"
                        .format(host, port, last_error)
                    )
                time.sleep(min(0.05, _remaining(deadline)))
        self.socket.settimeout(self.timeout)
        self.stream = self.socket.makefile("rwb")
        self.lock = threading.Lock()
        self.request({"op": "hello"})

    def request(self, request):
        with self.lock:
            self.stream.write((json.dumps(request) + "\n").encode("utf8"))
            self.stream.flush()
            line = self.stream.readline()
        if not line:
            raise ConnectionError("TCPStore server closed the connection")
        response = json.loads(line.decode("utf8"))
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "TCPStore request failed"))
        return response.get("result")

    def close(self):
        self.stream.close()
        self.socket.close()


class TCPStore(Store):
    """A small cross-process TCP key/value store.

    With no endpoint and a single worker it retains the useful local Store
    behavior expected by import-only compatibility probes.
    """

    def __init__(self, host_name=None, port=None, world_size=None,
                 is_master=False, timeout=None, wait_for_workers=True,
                 *args, **kwargs):
        size = 1 if world_size is None else int(world_size)
        if host_name is None and port is None and size == 1:
            Store.__init__(self, timeout)
            self.host = None
            self.port = None
            self._client = None
            self._server = None
            return
        if not host_name or port is None:
            raise ValueError("multi-process TCPStore requires host_name and port")
        self._timeout = _timeout_seconds(timeout)
        self.host = str(host_name)
        self.port = int(port)
        self._server = (
            _TCPStoreServer(self.host, self.port, self._timeout)
            if is_master else None
        )
        self._client = _TCPStoreClient(self.host, self.port, self._timeout)
        if wait_for_workers:
            self._client.request({
                "op": "wait_workers", "count": size, "timeout": self._timeout,
            })

    def _local(self):
        return self._client is None

    def set_timeout(self, timeout):
        self._timeout = _timeout_seconds(timeout)
        if self._local():
            Store.set_timeout(self, timeout)
        else:
            self._client.timeout = self._timeout
            self._client.socket.settimeout(self._timeout)
            if self._server is not None:
                self._server.timeout = self._timeout
                self._server.store.set_timeout(self._timeout)

    def set(self, key, value):
        if self._local():
            return Store.set(self, key, value)
        return self._client.request({
            "op": "set", "key": _key_text(key),
            "value": base64.b64encode(_value_bytes(value)).decode("ascii"),
        })

    def get(self, key):
        if self._local():
            return Store.get(self, key)
        encoded = self._client.request({
            "op": "get", "key": _key_text(key), "timeout": self._timeout,
        })
        return base64.b64decode(encoded)

    def add(self, key, num):
        if self._local():
            return Store.add(self, key, num)
        return int(self._client.request({
            "op": "add", "key": _key_text(key), "value": int(num),
        }))

    def wait(self, keys, timeout=None):
        if self._local():
            return Store.wait(self, keys, timeout)
        return self._client.request({
            "op": "wait", "keys": [_key_text(key) for key in keys],
            "timeout": self._timeout if timeout is None else _timeout_seconds(timeout),
        })

    def check(self, keys):
        if self._local():
            return Store.check(self, keys)
        return bool(self._client.request({
            "op": "check", "keys": [_key_text(key) for key in keys],
        }))

    def delete_key(self, key):
        if self._local():
            return Store.delete_key(self, key)
        return bool(self._client.request({
            "op": "delete", "key": _key_text(key),
        }))

    def num_keys(self):
        if self._local():
            return Store.num_keys(self)
        return int(self._client.request({"op": "num_keys"}))

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._server is not None:
            self._server.close()
            self._server = None


class FileStore(Store):
    """A cross-process store backed by one SQLite database file."""

    def __init__(self, file_name=None, world_size=None, timeout=None,
                 *args, **kwargs):
        if not file_name:
            raise ValueError("FileStore requires a file_name")
        self.path = os.fspath(file_name)
        self.world_size = 1 if world_size is None else int(world_size)
        self._timeout = _timeout_seconds(timeout)
        self._connection = sqlite3.connect(
            self.path, timeout=self._timeout, isolation_level=None,
            check_same_thread=False,
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS store (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        self._lock = threading.Lock()

    def set_timeout(self, timeout):
        self._timeout = _timeout_seconds(timeout)

    def set(self, key, value):
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO store(key, value) VALUES (?, ?)",
                (_key_text(key), sqlite3.Binary(_value_bytes(value))),
            )

    def _read(self, key):
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM store WHERE key = ?", (_key_text(key),)
            ).fetchone()
        return None if row is None else bytes(row[0])

    def get(self, key):
        deadline = _deadline(self._timeout)
        while True:
            value = self._read(key)
            if value is not None:
                return value
            if not _remaining(deadline):
                raise TimeoutError(
                    "timed out waiting for FileStore key {!r}".format(
                        _key_text(key))
                )
            time.sleep(min(0.01, _remaining(deadline)))

    def add(self, key, num):
        name = _key_text(key)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT value FROM store WHERE key = ?", (name,)
                ).fetchone()
                value = int(bytes(row[0])) if row is not None else 0
                value += int(num)
                self._connection.execute(
                    "INSERT OR REPLACE INTO store(key, value) VALUES (?, ?)",
                    (name, sqlite3.Binary(str(value).encode("ascii"))),
                )
                self._connection.execute("COMMIT")
                return value
            except (sqlite3.Error, TypeError, ValueError):
                self._connection.execute("ROLLBACK")
                raise

    def wait(self, keys, timeout=None):
        names = [_key_text(key) for key in keys]
        deadline = _deadline(self._timeout if timeout is None else timeout)
        while not self.check(names):
            if not _remaining(deadline):
                raise TimeoutError(
                    "timed out waiting for FileStore keys {}".format(names)
                )
            time.sleep(min(0.01, _remaining(deadline)))

    def check(self, keys):
        names = [_key_text(key) for key in keys]
        if not names:
            return True
        placeholders = ",".join("?" for _ in names)
        with self._lock:
            count = self._connection.execute(
                "SELECT COUNT(*) FROM store WHERE key IN ({})".format(placeholders),
                names,
            ).fetchone()[0]
        return count == len(set(names))

    def delete_key(self, key):
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM store WHERE key = ?", (_key_text(key),)
            )
        return cursor.rowcount > 0

    def num_keys(self):
        with self._lock:
            return int(self._connection.execute(
                "SELECT COUNT(*) FROM store"
            ).fetchone()[0])

    def close(self):
        self._connection.close()


class PrefixStore(Store):
    def __init__(self, prefix, store):
        self.prefix = str(prefix)
        self.store = store

    def _key(self, key):
        return self.prefix + _key_text(key)

    def set_timeout(self, timeout):
        return self.store.set_timeout(timeout)

    def set(self, key, value):
        return self.store.set(self._key(key), value)

    def get(self, key):
        return self.store.get(self._key(key))

    def add(self, key, num):
        return self.store.add(self._key(key), num)

    def wait(self, keys, timeout=None):
        return self.store.wait([self._key(key) for key in keys], timeout)

    def check(self, keys):
        return self.store.check([self._key(key) for key in keys])

    def delete_key(self, key):
        return self.store.delete_key(self._key(key))

    def num_keys(self):
        return self.store.num_keys()


def _resolved_int(explicit, query, environment, default):
    if int(explicit) >= 0:
        return int(explicit)
    if query:
        return int(query[0])
    value = os.environ.get(environment)
    return default if value is None else int(value)


def rendezvous(url, rank=-1, world_size=-1, timeout=None):
    """Yield ``(store, rank, world_size)`` for env/tcp/file URLs."""

    parsed = urlparse(str(url or "env://"))
    query = parse_qs(parsed.query)
    resolved_rank = _resolved_int(rank, query.get("rank"), "RANK", 0)
    resolved_world = _resolved_int(
        world_size, query.get("world_size"), "WORLD_SIZE", 1
    )
    if resolved_world < 1 or not 0 <= resolved_rank < resolved_world:
        raise ValueError(
            "rank {} is outside world size {}".format(
                resolved_rank, resolved_world)
        )

    if parsed.scheme == "env":
        host = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")
        if resolved_world == 1 and (not host or port is None):
            store = TCPStore(timeout=timeout)
        else:
            if not host or port is None:
                raise ValueError(
                    "env:// requires MASTER_ADDR and MASTER_PORT"
                )
            store = TCPStore(
                host, int(port), resolved_world, resolved_rank == 0,
                timeout=timeout,
            )
    elif parsed.scheme == "tcp":
        if not parsed.hostname or parsed.port is None:
            raise ValueError("tcp:// rendezvous requires host and port")
        store = TCPStore(
            parsed.hostname, parsed.port, resolved_world, resolved_rank == 0,
            timeout=timeout,
        )
    elif parsed.scheme == "file":
        store = FileStore(parsed.path, resolved_world, timeout=timeout)
    else:
        raise ValueError(
            "unsupported rendezvous scheme {!r}".format(parsed.scheme)
        )
    yield store, resolved_rank, resolved_world
