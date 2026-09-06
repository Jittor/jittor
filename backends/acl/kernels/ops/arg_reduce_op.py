import jittor as jt


class ArgReduceACL(jt.Function):
    def __init__(self, origin):
        super().__init__()
        self.origin = origin

    def execute(self, input, op, dim, keepdims=False):
        if dim < 0:
            dim += input.ndim
        self.input_shape = tuple(input.shape)
        self.dim = dim
        indices, values = self.origin(input, op, dim, keepdims)
        self.indices = indices
        return indices, values

    def grad(self, grad_indices, grad_values):
        if grad_values is None:
            return None, None, None, None

        scatter_shape = list(self.input_shape)
        scatter_shape[self.dim] = 1
        indices = self.indices.reshape(scatter_shape)
        source = grad_values.reshape(scatter_shape)
        grad_input = jt.scatter(
            jt.zeros(self.input_shape, dtype=grad_values.dtype),
            self.dim,
            indices,
            source,
            reduce="add",
        )
        return grad_input, None, None, None
