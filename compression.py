import torch
from memory import Memory

class SGDCompressor:
    def __init__(self, compress_ratio, memory=None,
                 warmup_epochs=-1, warmup_coeff=None):

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1

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

    def warmup_compress_ratio(self, epoch):
        # todo: change warm up op
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                         self.base_compress_ratio)
            else:
                compress_ratio = self.base_compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            self.compress_ratio = compress_ratio
            self.initialize(self.attributes.items())

    def _sparsify(self, tensor, name, percentage=0.01):
        tensor = tensor.view(-1)
        numel, shape = self.attributes[name]
        importance = tensor.abs()
        threshold = torch.min(torch.topk(importance, int(numel * percentage), 0, largest=True, sorted=False)[0])

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
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)

            grad.zero_().index_put_([indices], values, accumulate=True)

            return grad.view(shape)
        else:
            return self.memory.compensate(tensor, name, accumulate=False)

