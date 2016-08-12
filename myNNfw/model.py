# coding:utf-8

from layers import Layer


class Model(Layer):

    def __init__(self):
        super(Model, self).__init__(0, 0)
        self.layer_stack = []

    def add(self, layer):
        # 设置模型的输入输出
        # 第一层
        if len(self.layer_stack) == 0:
            self.input = layer.input
            self.output = layer.output
            self.input_dim = layer.input_dim
            self.output_dim = layer.output_dim
        # 不是第一层,只需设置输出即可
        else:
            self.output = layer.output
            self.output_dim = layer.output_dim

        # 设置层之间的连接，第一层的话不需要连接
        if len(self.layer_stack) != 0:
            # 把最上层的输出，赋给新层的输入
            layer.input = self.layer_stack[-1].output

        # 把新层加入新堆栈
        self.layer_stack.append(layer)

    def build(self, loss=None, optimizer=None):
        result = self.input
        # 逐层计算输出
        for layer in self.layer_stack:
            result = layer.call(result)
        self.loss = loss
        self.optimizer = optimizer

    def train(self):
        pass
