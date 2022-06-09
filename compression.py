import torch
from memory import Memory

class SGDCompressor:
    def __init__(self, compress_ratio=0.01, memory=None):
        self.compress_ratio = compress_ratio
        self.memory = Memory if memory is None else memory

        self.attributes = {}

    def initialize(self, named_parameters):
        """
        This function is actually of no use for us,
        only to keep consistency with dgc.
        :param named_parameters:
        :return:
        """
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]

            self.attributes[name] = (numel, shape)

    def _sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape = self.attributes[name]
        importance = tensor.abs()
        threshold = torch.min(torch.topk(importance, int(numel * self.compress_ratio), 0, largest=True, sorted=False)[0])

        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)

        values = tensor[indices]
        return values, indices, numel, shape

    def compress(self, tensor, name):
        if name in self.attributes:
            # compress
            tensor_compensated = self.memory.compensate(
                tensor, name, accumulate=True)
            values, indices, numel, shape = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))

            return (values, indices), ctx
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)

            grad.zero_().index_put_([indices], values, accumulate=True)

            return grad.view(shape)
        else:
            return self.memory.compensate(tensor, name, accumulate=False)

