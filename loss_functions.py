import numpy as np
import activation_functions as afs

class LossFunction():

    def jacobian(self):
        raise NotImplementedError()

    def hessian(self):
        raise NotImplementedError()

class LeastSquares(LossFunction):

    def jacobian(self):
        return True

    def hessian(self):
        return True