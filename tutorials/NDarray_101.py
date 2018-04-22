'''
This is the tutorials for NDarray
'''

import mxnet as mx
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pk
import gc
from memory_profiler import profile
from memory_profiler import memory_usage

'''
First create NDarray from mxnet
We can create ndarray containing in the gpu or cpu by adding mx.gpu or mx.cpu.
By default, the arrays are in the cpu
'''
@profile
#create a ndarray from mx containing in the gpu
def fun():
    a=mx.nd.array([1,2,3],mx.gpu())
    b=mx.nd.array(([1,2,3],[2,3,4]),mx.gpu())
    c=mx.nd.array([1,2,3])

    print 'NDarray: a,',a,' b,',b,' c',c

    #create a ndarray from numpy

    d=np.arange(10).reshape((2,-1))
    e=mx.nd.array(d,dtype=np.float32).as_in_context(mx.gpu())
    f=np.random.random((100,100,100,100))

    print 'numpy array,',d,' mx array from numpy', e

    # save data to disk by using pickle
    a=None
    b=None
    c=None
    d=None
    e=None
    f=None
    gc.collect()

    a=mx.nd.ones((2,3),mx.gpu())
    data=pk.dumps(a)
    pk.dump(data,open('temp.pickle','wb'))
    a=None
    data=None
    gc.collect()

    # load from disk using pickle

    b=pk.load(open('temp.pickle','rb'))
    data=pk.loads(b)
    print data
    b=None
    data=None
    gc.collect()

    # the other way to save data is using save or load

    a=mx.nd.random_normal(shape=(5,5,5),ctx=mx.gpu(),dtype=np.float32)
    mx.nd.save('temp.ndarray',a)
    a=None
    gc.collect()
    a=mx.nd.load('temp.ndarray')
    print a

if __name__ == '__main__':
    fun()
    print memory_usage(-1, interval=0.2, timeout=1)