import numpy as np
from step_12 import Function


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")

        self.data = data 
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)
            
        return True


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y 

    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data 
        gx = 2*x * gy 
        return gx 


def add(x0, x1):
    return Add()(x0, x1)


def square(x0):
    return Square()(x0)


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))

    print(x.data, y.data, z.data)