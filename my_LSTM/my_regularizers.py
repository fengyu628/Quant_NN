# coding:utf-8

import my_theano_backend as K


class Regularizer(object):

    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {'name': self.__class__.__name__}


class WeightRegularizer(Regularizer):

    # def __init__(self, l1=0., l2=0.):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True
        self.p = None

    def set_param(self, p):
        if self.p is not None:
            raise Exception('Regularizers cannot be reused. '
                            'Instantiate one regularizer per layer.')
        self.p = p

    def __call__(self, loss):
        if self.p is None:
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss
        if self.l1:
            regularized_loss += K.sum(self.l1 * K.abs(self.p))
        if self.l2:
            regularized_loss += K.sum(self.l2 * K.square(self.p))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}
