import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

class MemoryState(object):
    def __init__(self, process_group, momentum=0.9, nesterov=False,
                 start_iter=0, compress_ratio=0.01,
                 gradient_clipping=None, momentum_masking=True):
        self.process_group = process_group
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.compress_ratio = compress_ratio

        self.momentum = momentum
        self.nesterov = nesterov

        self.momentums = []
        self.velocities = []
        self.numels = []

        self.iter = 0
        self.start_iter = start_iter

    def initialize(self, parameters):
        for param in parameters:
            self.momentums.append(torch.zeros_like(param.data))
            self.velocities.append(torch.zeros_like(param.data))
            self.numels.append(param.numel())

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_the_last_bucket_to_allreduce():
            self.iter += 1

    def compensate(self, grad, index, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[index]
        if accumulate:
            vec = self.velocities[index]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()


def compress_hook(state, bucket):
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensor()

    # Run vanilla allreduce in the first `start_iter` iterations.
    if state.iter < state.start_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.get_per_parameter_tensors()

    all_indices, all_values = torch.tensor([]), torch.tensor([])
    start_idx, end_idx = 0, 0
    for i, tensor in enumerate(tensors):
        start_idx = end_idx
        end_idx = start_idx + state.numels[i]
        # update memory state with the new grad tensor
        state.compensate(tensor, i)

        # find the compression threshold
        tensor = tensor.view(-1)
        importance = tensor.abs()
        threshold = torch.min(torch.topk(importance, int(importance.numel() * state.compress_ratio), 0, largest=True, sorted=False)[0])

        # get indices to keep for this process
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        values = tensor[indices]

        # clear corresponding gradients
        if state.momentum_masking:
            state.momentums[i].view(-1).index_fill_(0, indices, 0)
        state.velocities[i].view(-1).index_fill_(0, indices, 0)

        # map to input_tensor indices
        indices.add_(start_idx)

        all_indices = torch.cat((all_indices, indices))
        all_values = torch.cat((all_values, values))

    ctx = torch.stack((all_indices, all_values), 0)

    # communication
    out_list = [torch.zeros_like(ctx, device=ctx.device,
            dtype=ctx.dtype) for _ in range(world_size)]

    fut = dist.all_gather(
        out_list, ctx, group=group_to_use, async_op=True).get_future()

    def decompress(fut):
        agg_tensor = fut.value()[0]
        indices, values = agg_tensor[0], agg_tensor[1]
        output_tensor = torch.zeros_like(input_tensor, device=input_tensor.device, dtype=input_tensor.dtype)
        output_tensor[indices] = values

        return [output_tensor]
    return fut.then(decompress)














