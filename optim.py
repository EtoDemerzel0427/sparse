# modified from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

import torch
from torch.optim.optimizer import Optimizer #, required

__all__ = ['SparseSGD']

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class SparseSGD(Optimizer):
    def __init__(self, params, named_parameters, lr=required, compressor=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.compressor = compressor
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [('allreduce.noname.%s' % i, v)
                        for param_group in self.param_groups
                        for i, v in enumerate(param_group['params'])]

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.warmup = True
        super(SparseSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SparseSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def synchronize(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                name = self._parameter_names.get(p)
                tensor_compressed, ctx = self.compressor.compress(p.grad, name)

                # fake synchronization, decompress in-place
                #p.grad.set_(self.compressor.decompress(tensor_compressed, ctx))
                #print(f"Nonzeros before:{torch.count_nonzero(p.grad)}")
                p.grad = self.compressor.decompress(tensor_compressed, ctx)
                #print(f"Nonzeros after:{torch.count_nonzero(p.grad)}")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.warmup:
            self.synchronize()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay != 0:
                    d_p = weight_decay * p.data
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    d_p = d_p.add(p.grad)
                else:
                    d_p = p.grad

                p.add_(d_p, alpha=-group['lr'])

        return loss
