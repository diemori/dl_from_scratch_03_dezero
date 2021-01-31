import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 2 else outputs[0]


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

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            print(f"gys: {gys}, gxs: {gxs}")

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

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
    x = Variable(np.array(3.0))
    k = Variable(np.array(2.0))
    y = add(add(x, k), x)

    print(f"y.data: {y.data}")

    y.backward()
    print(f"x.grad: {x.grad}")