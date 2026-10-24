"""Node-backed jagged shape metadata for the Torch compatibility frontend.

Supports identity/coefficient metadata only, not native symbolic allocation or
expression graphs. Runtime identity caches carry no Tensor/Var/Op references.
"""

from weakref import WeakKeyDictionary
from uuid import uuid4 as _uuid4


class _NestedSymbolOwner:
    """Weak-keyable sentinel; it has no link to the Runtime or installed module."""
    __slots__ = ('identity', '__weakref__')

    def __init__(self):
        # OS-backed identity only; no Python/Torch/Jittor RNG state is consumed.
        self.identity = _uuid4().bytes

    def __reduce_ex__(self, protocol):
        # Copying an owner creates a NEW owner; terms copy only identity values.
        return type(self), ()


class _LegacyNestedSymbolCache:
    """Holder-owned weak-owner cache, never persistent Tensor state.

    Standard copy/pickle must reset this runtime-only cache rather than try to
    serialize weakrefs or transfer an original holder's symbolic identity.
    """
    __slots__ = ('_values',)

    def __init__(self):
        self._values = WeakKeyDictionary()

    def get(self, owner):
        return self._values.get(owner)

    def put(self, owner, value):
        self._values[owner] = value

    def __reduce_ex__(self, protocol):
        return type(self), ()


class _NestedSymbolToken(tuple):
    """Immutable serializable logical identity, with no owner/Tensor back-link."""
    __slots__ = ()

    def __new__(cls, owner_identity, ordinal):
        if type(owner_identity) is not bytes or len(owner_identity) != 16:
            raise TypeError('owned opaque identity bytes required')
        if type(ordinal) is not int or ordinal < 0:
            raise TypeError('nested symbol ordinal must be a nonnegative integer')
        return tuple.__new__(cls, (owner_identity, ordinal))

    @property
    def ordinal(self):
        return self[1]

    def __getnewargs__(self):
        return self[0], self[1]

    def nested_int(self):
        # An injective pairing of the opaque owner value and unbounded ordinal.
        # This is metadata identity, never a concrete allocation dimension.
        owner = int.from_bytes(self[0], 'big')
        total = owner + self[1]
        return total * (total + 1) // 2 + self[1]


class _ConstantBoolNode:
    __slots__ = ('value',)

    def __init__(self, value):
        if type(value) is not bool:
            raise TypeError('constant bool node requires bool')
        self.value = value

    def is_int(self): return False
    def is_float(self): return False
    def is_bool(self): return True
    def is_constant(self): return True
    def guard_bool(self, file, line): return self.value


class _ConstantIntNode:
    __slots__ = ('value',)

    def __init__(self, value):
        if type(value) is not int:
            raise TypeError('constant int node requires int')
        self.value = value

    def is_int(self): return True
    def is_float(self): return False
    def is_bool(self): return False
    def is_constant(self): return True
    def constant_int(self): return self.value

    def mul(self, other):
        # Reverse public integer multiplication delegates to the same owner.
        # This is not a second expression/mathematical simplifier.
        if type(other) is _NestedIntNode:
            return other.mul(self)
        raise NotImplementedError('only integer times jagged metadata is supported')


def _eq(lhs, rhs):
    return (type(lhs) is _NestedIntNode and type(rhs) is _NestedIntNode
            and lhs.token == rhs.token and lhs.coeff == rhs.coeff)


def _ge(lhs, rhs):
    if type(lhs) is _NestedIntNode and type(rhs) is _NestedIntNode:
        if lhs.token == rhs.token:
            return lhs.coeff >= rhs.coeff
        raise ValueError('ge: relation is indeterminate')
    if type(lhs) is _NestedIntNode:
        if rhs.is_constant() and rhs.constant_int() <= 2:
            return True
        raise ValueError('ge: relation is indeterminate')
    if type(rhs) is _NestedIntNode:
        if lhs.is_constant() and lhs.constant_int() < 2:
            return False
        raise ValueError('ge: relation is indeterminate')
    raise ValueError('inputs unsupported')


class _NestedIntNode:
    """One immutable-identity token and integer coefficient, no expression graph."""
    __slots__ = ('token', 'coeff')

    def __init__(self, token, coeff=1):
        if type(token) is not _NestedSymbolToken or type(coeff) is not int:
            raise TypeError('owned nested token and integer coefficient required')
        self.token, self.coeff = token, coeff

    def nested_int_coeff(self): return self.coeff
    def maybe_as_int(self): return None
    def is_int(self): return True
    def is_float(self): return False
    def is_bool(self): return False
    def is_nested_int(self): return True
    def is_symbolic(self): return False
    def is_constant(self): return False
    def nested_int(self): return self.token.nested_int()
    def clone(self): return self

    def _str(self):
        label = 'j' + str(self.token.nested_int())
        return label if self.coeff == 1 else str(self.coeff) + '*' + label

    str = _str
    __str__ = _str
    __repr__ = _str
    _graph_repr = _str

    def wrap_int(self, value):
        return _ConstantIntNode(value)

    def mul(self, other):
        if not other.is_constant():
            raise ValueError('nested metadata multiplication requires a constant integer')
        return _NestedIntNode(self.token, self.coeff * other.constant_int())

    def eq(self, other): return _ConstantBoolNode(_eq(self, other))
    def ne(self, other): return _ConstantBoolNode(not _eq(self, other))
    def gt(self, other): return _ConstantBoolNode(not _ge(other, self))
    def lt(self, other): return _ConstantBoolNode(not _ge(self, other))
    def le(self, other): return _ConstantBoolNode(_ge(other, self))
    def ge(self, other): return _ConstantBoolNode(_ge(self, other))


def _to_node(node, value):
    if isinstance(value, SymInt):
        return value.node
    if type(value) is bool:
        return node.wrap_bool(value)  # Unsupported by jagged nodes, as in Native.
    if type(value) is int:
        return node.wrap_int(value)
    if type(value) is float:
        raise NotImplementedError('floating promotion of jagged metadata is not supported')
    return NotImplemented


def _wrap_result(node):
    if node.is_int():
        return SymInt(node)
    if type(node) is _ConstantBoolNode:
        return node.guard_bool('', 0)
    raise NotImplementedError('only jagged integer metadata and constant predicates are supported')


def _binary(self, other, method, reverse=False):
    other_node = _to_node(self.node, other)
    if other_node is NotImplemented:
        return NotImplemented
    left, right = (other_node, self.node) if reverse else (self.node, other_node)
    # Missing add/sub/int_ methods naturally fail at the protocol boundary;
    # no observed exception category or message is hardcoded here.
    return _wrap_result(getattr(left, method)(right))


class SymInt:
    """Public node-backed integer facade; only jagged metadata is implemented.

    Not an int subclass: ordinary integers never acquire a symbolic category.
    Native allocation/reshape must still reject nodes without concrete int_().
    """
    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __bool__(self): return bool(self != 0)
    def __int__(self): return self.node.int_()
    def __index__(self): return self.node.int_()
    def __round__(self, ndigits=None): return self
    def __repr__(self): return self.node._graph_repr()

    def __hash__(self):
        if self.node.is_nested_int():
            return hash(self.node.nested_int())
        raise TypeError('unhashable type: non-nested SymInt')

    def as_integer_ratio(self): return self, 1
    def bit_length(self): return int(self).bit_length()
    def __eq__(self, other): return _binary(self, other, 'eq')
    def __ne__(self, other): return _binary(self, other, 'ne')
    def __lt__(self, other): return _binary(self, other, 'lt')
    def __le__(self, other): return _binary(self, other, 'le')
    def __gt__(self, other): return _binary(self, other, 'gt')
    def __ge__(self, other): return _binary(self, other, 'ge')
    def __mul__(self, other): return _binary(self, other, 'mul')
    def __rmul__(self, other): return _binary(self, other, 'mul', reverse=True)
    def __add__(self, other): return _binary(self, other, 'add')
    def __radd__(self, other): return _binary(self, other, 'add', reverse=True)
    def __sub__(self, other): return _binary(self, other, 'sub')
    def __rsub__(self, other): return _binary(self, other, 'sub', reverse=True)
