from collections import namedtuple
from quantum_blur import blur_image
from myrtle_core import normalise, unnormalise
from myrtle_torch_backend import cifar10_mean, cifar10_std
import numpy as np


class QuantumBlur(namedtuple('QuantumBlur', ('h', 'w', 'alpha'))):
    
    def __call__(self, x, x0, y0):
        x_unnormalised = unnormalise(x[..., y0:y0+self.h, x0:x0+self.w], mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32))
        x[..., y0:y0+self.h, x0:x0+self.w] = normalise(blur_image(x_unnormalised,self.alpha).transpose((1,2,0)), mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)).transpose((2,0,1))
        return x

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]


class GaussianBlur(namedtuple('GaussianBlur', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x[..., y0:y0+self.h, x0:x0+self.w] = x[..., y0:y0+self.h, x0:x0+self.w] + np.random.normal(loc=0, scale=0.01, size=(3,self.h,self.w))
        return x

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]

