from MS326.layers import *
from MS326.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform, batch normalization and ReLU
    """
    out1, fc_cache = affine_forward(x, w, b)
    out2, bn_cache = batchnorm_forward(out1, gamma, beta, bn_param)
    out3, relu_cache = relu_forward(out2)
    cache = (fc_cache, bn_cache, relu_cache)
    return out3, cache


def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    d1 = relu_backward(dout, relu_cache)
    d2, dgamma, dbeta = batchnorm_backward(d1, bn_cache)
    d3, dw, db = affine_backward(d2, fc_cache)
    return d3, dw, db, dgamma, dbeta


def affine_ln_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform, layer normalization and ReLU
    """
    out1, fc_cache = affine_forward(x, w, b)
    out2, bn_cache = layernorm_forward(out1, gamma, beta, bn_param)
    out3, relu_cache = relu_forward(out2)
    cache = (fc_cache, bn_cache, relu_cache)
    return out3, cache


def affine_ln_relu_backward(dout, cache):
    """
    Backward pass for the affine-ln-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    d1 = relu_backward(dout, relu_cache)
    d2, dgamma, dbeta = layernorm_backward(d1, bn_cache)
    d3, dw, db = affine_backward(d2, fc_cache)
    return d3, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
