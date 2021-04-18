# coding : utf-8
import sys
import os
sys.path.append(os.pardir)
import cupy as cp
from dataset.mnist import MNIST
from twolayernet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = MNIST().get_dataset(flatten=False, normalize=True, one_hot=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = cp.asarray(x_train[:3])
t_batch = cp.asarray(t_train[:3])

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)
for key in grad_numerical.keys():
    diff = cp.average(cp.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))
    
