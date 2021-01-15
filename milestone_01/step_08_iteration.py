import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

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

        func = self.creator

        while func is not None:
            x, y = func.input, func.output 
            x.grad = func.backward(y.grad)

            func = x.creator

        return True


class Function:
    def __call__(self, input):
        x = input.data 
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input 
        self.output = output 
        return output 


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def example_8_3():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
    

if __name__ == "__main__":
    # y = example_7_1()
    # example_7_2(y)
    example_8_3()
