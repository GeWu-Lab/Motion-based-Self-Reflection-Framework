
import torch
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F


def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
            

def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of 
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be 
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))

def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions in a tensor to a target dimension.

    Args:
        x (torch.Tensor): tensor to reshape
        begin_axis (int): begin dimension
        end_axis (int): end dimension (inclusive)
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (torch.Tensor): reshaped tensor
    """
    assert(begin_axis <= end_axis)
    assert(begin_axis >= 0)
    assert(end_axis < len(x.shape))
    assert(isinstance(target_dims, (tuple, list)))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)

def join_dimensions(x, begin_axis, end_axis):
    """
    Joins all dimensions between dimensions (@begin_axis, @end_axis) into a flat dimension, for
    all tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]),
            np.ndarray: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]),
            type(None): lambda x: x,
        }
    )
    

def flatten_nested_dict_list(d, parent_key='', sep='_', item_key=''):
    """
    Flatten a nested dict or list to a list.

    For example, given a dict
    {
        a: 1
        b: {
            c: 2
        }
        c: 3
    }

    the function would return [(a, 1), (b_c, 2), (c, 3)]

    Args:
        d (dict, list): a nested dict or list to be flattened
        parent_key (str): recursion helper
        sep (str): separator for nesting keys
        item_key (str): recursion helper
    Returns:
        list: a list of (key, value) tuples
    """
    items = []
    if isinstance(d, (tuple, list)):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for i, v in enumerate(d):
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=str(i)))
        return items
    elif isinstance(d, dict):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for k, v in d.items():
            assert isinstance(k, str)
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=k))
        return items
    else:
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        return [(new_key, d)]

def map_tensor(x, func):
    """
    Apply function @func to torch.Tensor objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each tensor

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda x: x,
        }
    )

def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions for all tensors in nested dictionary or list or tuple 
    to a target dimension.
    
    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension (inclusive)
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t),
            np.ndarray: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t),
            type(None): lambda x: x,
        }
    )



def time_distributed(inputs, op, activation=None, inputs_as_kwargs=False, inputs_as_args=False, **kwargs):
    """
    Apply function @op to all tensors in nested dictionary or list or tuple @inputs in both the
    batch (B) and time (T) dimension, where the tensors are expected to have shape [B, T, ...].
    Will do this by reshaping tensors to [B * T, ...], passing through the op, and then reshaping
    outputs to [B, T, ...].

    Args:
        inputs (list or tuple or dict): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        op: a layer op that accepts inputs
        activation: activation to apply at the output
        inputs_as_kwargs (bool): whether to feed input as a kwargs dict to the op
        inputs_as_args (bool) whether to feed input as a args list to the op
        kwargs (dict): other kwargs to supply to the op

    Returns:
        outputs (dict or list or tuple): new nested dict-list-tuple with tensors of leading dimension [B, T].
    """
    batch_size, seq_len = flatten_nested_dict_list(inputs)[0][1].shape[:2]
    inputs = join_dimensions(inputs, 0, 1)
    if inputs_as_kwargs:
        outputs = op(**inputs, **kwargs)
    elif inputs_as_args:
        outputs = op(*inputs, **kwargs)
    else:
        outputs = op(inputs, **kwargs)

    if activation is not None:
        outputs = map_tensor(outputs, activation)
    outputs = reshape_dimensions(outputs, begin_axis=0, end_axis=0, target_dims=(batch_size, seq_len))
    return outputs

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)