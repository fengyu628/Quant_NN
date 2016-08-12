# coding:utf-8

from myNNfw import model
from myNNfw import layers

model = model.Model()
dense_1 = layers.Dense(input_dim=2, output_dim=2)
model.add(dense_1)
tanh_1 = layers.ActivationTanH(input_dim=2, output_dim=2)
model.add(tanh_1)
dense_2 = layers.Dense(input_dim=2, output_dim=1)
model.add(dense_2)
tanh_2 = layers.ActivationTanH(input_dim=1, output_dim=1)
model.add(tanh_2)
# print(model.layer_stack)
