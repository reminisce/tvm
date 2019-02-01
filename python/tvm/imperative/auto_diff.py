import numpy as np
import tvm
import topi


class TopoOrderedNode:
    def __init__(self, node, output_entries):
        self.node = node
        self.output_entries = output_entries


class AutoDiffRecorder:
    target = tvm.target.create('llvm')
    cached_ndarrays = {}  # dict: NodeEntry -> NDArray
    num_nodes = 0
    grad_func_map = {}
    topo_ordered_nodes = []
    node_idx_map = {}

    @staticmethod
    def record_op(op, attrs, inputs, outputs):
        node = Node(op, attrs, [array.node_entry for array in inputs])
        node.num_outputs = len(outputs)
        for i, output in enumerate(outputs):
            output.node_entry = NodeEntry(node, i)
            AutoDiffRecorder.cache_ndarray(output)

        for input in inputs:
            AutoDiffRecorder.cache_ndarray(input)

        AutoDiffRecorder.node_idx_map[node] = len(AutoDiffRecorder.topo_ordered_nodes)
        AutoDiffRecorder.topo_ordered_nodes.append(TopoOrderedNode(node, [output.node_entry for output in outputs]))

    @staticmethod
    def cache_ndarray(array):
        if array.node_entry not in AutoDiffRecorder.cached_ndarrays:
            AutoDiffRecorder.cached_ndarrays[array.node_entry] = array

    @staticmethod
    def backprop(node_entry):
        if node_entry.node.op is None:
            raise ValueError("node_entry = {} is a variable node output".format(node_entry.__repr__()))
        if node_entry.node not in AutoDiffRecorder.node_idx_map:
            raise ValueError('node = {} is not recorded'.format(node_entry.node.__repr__()))
        idx = AutoDiffRecorder.node_idx_map.get(node_entry.node)
        output = AutoDiffRecorder.cached_ndarrays.get(node_entry, None)
        assert output is not None
        output.grad = NDArray(np.ones_like(output.array.asnumpy()))
        for topo_ordered_node in reversed(AutoDiffRecorder.topo_ordered_nodes[0:idx+1]):
            assert topo_ordered_node.node.op is not None
            inputs = [AutoDiffRecorder.cached_ndarrays.get(entry) for entry in topo_ordered_node.node.input_entries]
            outputs = [AutoDiffRecorder.cached_ndarrays.get(entry) for entry in topo_ordered_node.output_entries]
            diff_op = AutoDiffRecorder.grad_func_map.get(topo_ordered_node.node.op.__name__, None)
            if diff_op is None:
                raise ValueError('Operator {} has not registered diff function'
                                 .format_map(topo_ordered_node.node.op.__name__))
            diff_op(inputs, outputs, **topo_ordered_node.node.attrs)


class Node:
    def __init__(self, op=None, attrs=None, input_entries=[]):
        self.op = op
        self.attrs = attrs
        self.input_entries = input_entries
        self.output_entries = []
        if op is None:
            self.name = 'variable'
        else:
            self.name = op.__name__
        self.name += '_' + str(AutoDiffRecorder.num_nodes)
        AutoDiffRecorder.num_nodes += 1

    def is_variable(self):
        return self.op is None

    def __repr__(self):
        return self.name

    @property
    def num_inputs(self):
        return len(self.input_entries)


class NodeEntry:
    def __init__(self, node=Node(), index=0):
        self.node = node
        self.index = index

    def __hash__(self):
        return hash(self.node) ^ hash(self.index)

    def __eq__(self, other):
        return self.node == other.node and self.index == other.index

    def __repr__(self):
        return "<NodeEntry: node={}, index={}>\n".format(self.node.__repr__(), self.index)


class NDArray(object):
    def __init__(self, array, node_entry=None):
        if isinstance(array, np.ndarray):
            self.array = tvm.ndarray.array(array)
        elif isinstance(array, tvm.ndarray.NDArray):
            self.array = array
        else:
            raise NotImplemented
        self.node_entry = node_entry
        self.grad = None

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __repr__(self):
        return self.array.__repr__()

    def mark_variable(self):
        if self.node_entry is not None:
            raise NotImplemented
        self.node_entry = NodeEntry()

    def backward(self):
        AutoDiffRecorder.backprop(self.node_entry)


def sum(data, axis=None, keepdims=False, out=None):
    data_tvm = tvm.placeholder(shape=data.shape, dtype=data.dtype)
    out_tensor = topi.sum(data_tvm, axis=axis, keepdims=keepdims)
    with AutoDiffRecorder.target:
        s = topi.generic.schedule_reduce(out_tensor)
    if out is None:
        out = tvm.ndarray.empty(shape=out_tensor.shape, dtype=out_tensor.dtype)
    func = tvm.build(s, [data_tvm, out_tensor])
    func(data.array, out)
    out = NDArray(out)
    AutoDiffRecorder.record_op(sum, {'axis': axis, 'keepdims': keepdims}, [data], [out])
    return out


def diff_sum(inputs, outputs, axis=None, keepdims=False):
    assert len(inputs) == 1
    assert len(outputs) == 1
    input_data = inputs[0]
    output_data = outputs[0]
    if output_data.grad is None:
        return
    output_grad = output_data.grad
    input_data_tensor = tvm.placeholder(shape=input_data.shape, dtype=input_data.dtype)
    output_data_tensor = topi.sum(input_data_tensor, axis=axis, keepdims=keepdims)
    head = tvm.placeholder(shape=output_grad.shape, dtype=output_grad.dtype)
    [d_input_data_tensor] = tvm.autodiff.differentiate(output_data_tensor, [input_data_tensor], head)
    with AutoDiffRecorder.target:
        s = topi.generic.schedule_injective(d_input_data_tensor)
    func = tvm.build(s, [d_input_data_tensor, head])
    d_input_data = tvm.nd.array(np.zeros_like(input_data.array.asnumpy()))
    func(d_input_data, output_grad.array)
    if input_data.grad is None:
        input_data.grad = NDArray(d_input_data)
    else:
        tmp = input_data.grad.array.asnumpy()
        tmp += d_input_data.asnumpy()
        input_data.grad = NDArray(tmp)


AutoDiffRecorder.grad_func_map['sum'] = diff_sum


data = tvm.ndarray.array(np.ones((2, 3, 4)).astype('float32'))
data = NDArray(data)
data.mark_variable()
out = sum(data, axis=1)
#diff_sum([data], [out], axis=1)
out = sum(out, axis=0)
out = sum(out, axis=0)
print('----------final output----------')
print(out)
out.backward()
print('\n')
print('----------data.grad----------')
print(data.grad)


for k, v in AutoDiffRecorder.cached_ndarrays.items():
    print('=====================================')
    print(k)
    print(v)
