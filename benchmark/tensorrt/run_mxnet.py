import os
import time
import numpy as np
import argparse
from mxnet.gluon.model_zoo.vision import get_model
import mxnet as mx

batch_size = 1

models = ['resnet18_v1',
          'resnet34_v1',
          'resnet50_v1',
          'resnet101_v1',
          'resnet152_v1',
          'resnet18_v2',
          'resnet34_v2',
          'resnet50_v2',
          'resnet101_v2',
          'resnet152_v2',
          'vgg11',
          'vgg13',
          'vgg16',
          'vgg19',
          'vgg11_bn',
          'vgg13_bn',
          'vgg16_bn',
          'vgg19_bn',
          'alexnet',
          'densenet121',
          'densenet161',
          'densenet169',
          'densenet201',
          'squeezenet1.0',
          'squeezenet1.1',
          'inceptionv3',
          'mobilenet1.0',
          'mobilenet0.75',
          'mobilenet0.5',
          'mobilenet0.25',
          'mobilenetv2_1.0',
          'mobilenetv2_0.75',
          'mobilenetv2_0.5',
          'mobilenetv2_0.25']


def get_data_shape(model_name):
    if model_name.startswith('inception'):
        return (batch_size, 3, 299, 299)
    else:
        return (batch_size, 3, 224, 224)


def get_mxnet_workload(network, **kwargs):
    block = get_model(network, **kwargs)
    block.hybridize()
    block.forward(mx.nd.zeros(data_shape))
    block.export(network)
    return mx.model.load_checkpoint(network, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark MXNet')
    parser.add_argument('--network', type=str, required=True, choices=models)
    args = parser.parse_args()

    network = args.network
    num_classes = 1000
    sym, arg_params, aux_params = get_mxnet_workload(network, pretrained=True)
    data_shape = get_data_shape(network)

    # Execute with MXNet
    os.environ['MXNET_USE_TENSORRT'] = '0'
    executor = sym.simple_bind(ctx=mx.gpu(0), data=data_shape, grad_req='null', force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)

    input = np.random.uniform(size=data_shape).astype("float32")
    # Warmup
    repeat = 100
    print('=============Warming up MXNet...')
    for i in range(repeat):
        y_gen = executor.forward(is_train=False, data=input)
        y_gen[0].wait_to_read()

    # Timing
    print('=============Starting MXNet timed run...')
    start = time.time()
    for i in range(0, repeat):
        y_gen = executor.forward(is_train=False)
        y_gen[0].wait_to_read()
    end = time.time()
    elapse = (time.time() - start) * 1000.0 / repeat
    print("MXNet w/o TensorRT runtime per forward: %sms" % elapse)
    print("MXNet w/o TensorRT throughput: %d images/s" % int(1000. / elapse))

    # Execute with TensorRT
    print('=============Building TensorRT engine...')
    os.environ['MXNET_USE_TENSORRT'] = '1'
    arg_params.update(aux_params)
    all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in arg_params.items()])
    executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(0), all_params=all_params,
                                                 data=data_shape, grad_req='null', force_rebind=True)
    # Warmup
    print('=============Warming up TensorRT')
    for i in range(repeat):
        y_gen = executor.forward(is_train=False, data=input)
        y_gen[0].wait_to_read()

    # Timing
    print('============Starting TensorRT timed run')
    start = time.time()
    for i in range(0, repeat):
        y_gen = executor.forward(is_train=False)
        y_gen[0].wait_to_read()
    elapse = (time.time() - start) * 1000.0 / repeat
    print("MXNet w/ TensorRT runtime per forward: %sms" % elapse)
    print("MXNet w/ TensorRT throughput: %d images/s" % int(1000. / elapse))

