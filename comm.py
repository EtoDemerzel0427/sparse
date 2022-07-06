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

        # dict keys are bucket id
        self.momentum_dict = {}
        self.volocities_dict = {}

        self.iter = 0
        self.start_iter = start_iter

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_the_last_bucket_to_allreduce():
            self.iter += 1


def compress_hook(state: MemoryState, bucket):
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensor()

    # Run vanilla allreduce in the first `start_iter` iterations.
    if state.iter < state.start_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    device = input_tensor.device
    dtype = input_tensor.dtype

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.get_per_parameter_tensors()

    # divide tensors into two groups,
    # one will be compressed before doing all_gather, another will directly do allreduce as normal without compression
    tensors_to_compress, uncompressed_tensors = [], []
    compress_size, uncompress_size = 0, 0
    for tensor in tensors:
        # don't compress tensors with too few parameters
        if tensor.numel() * state.compress_ratio < 1.:
            uncompressed_tensors.append(tensor)
            uncompress_size += tensor.numel()
        else:
            tensors_to_compress.append(tensor)
            compress_size += tensor.numel()

    bucket_id = bucket.get_index()
    if bucket_id not in state.momentum_dict:
        state.momentum_dict[bucket_id] = torch.zeros(compress_size, device=device, dtype=dtype)
        state.volocities_dict[bucket_id] = torch.zeros(uncompress_size, device=device, dtype=dtype)

    # handle uncompressed_tensors
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # handle the tensors that should be compressed
    all_indices, all_values = [], []
    start_idx = 0
    for tensor in tensors_to_compress:
        # accumulate local gradients with current grad tensor adding in
        mmt = state.momentum_dict[bucket_id][start_idx: start_idx + tensor.numel()]
        vec = state.volocities_dict[bucket_id][start_idx: start_idx + tensor.numel()]
        if state.nesterov:
            mmt.add_(tensor.view(-1)).mul_(state.momentum)
            vec.add_(mmt).add_(tensor.view(-1))
        else:
            mmt.mul_(state.momentum).add_(tensor.view(-1))
            vec.add_(mmt)

        # select top k
        importance = tensor.abs().view(-1)
        threshold = torch.min(
            torch.topk(importance, int(importance.numel() * state.compress_ratio), 0, largest=True, sorted=False)[0])
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        values = mmt.clone()[indices]

        # map indices to global
        indices.add_(start_idx)

        # clear those entries in memory states
        state.momentum_dict[bucket_id].index_fill_(0, indices, 0)
        state.volocities_dict[bucket_id].index_fill_(0, indices, 0)

        all_indices.append(indices)
        all_values.append(values)

        start_idx += tensor.numel()

    all_indices = torch.cat(all_indices)
    all_values = torch.cat(all_values)
    output_indices = [torch.zeros_like(all_indices, device=device, dtype=dtype) for _ in world_size]
    output_values = [torch.zeros_like(all_values, device=device, dtype=dtype) for _ in world_size]

    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_uncompressed_tensors_and_allgather_indices(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        return [
            dist.all_gather(
                output_indices, all_indices, group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def unpack_indices_and_all_gather_values(fut):
        global output_indices
        output_indices = fut.value()[0]

        return [
            dist.all_gather(
            output_values, all_values, group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        output_values = fut.value()[0]

        for tensor in tensors_to_compress:
            torch.zeros(tensor.shape, out=tensor)

        for (indices, values) in zip(output_indices, output_values):
            input_tensor[indices].add_(values)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        input_tensor.div_(world_size)

        state.maybe_increase_iter(bucket)

        return [input_tensor]

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_indices_and_all_gather_values
        )
        .then(unpack_indices_and_all_gather_values)
        .then(decompress)
    )




