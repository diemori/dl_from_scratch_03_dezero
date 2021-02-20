# step 22: operator overload 3

import numpy as np
import weakref
import contextlib


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


class Config:
    enable_backprop = True


class Function:
    def __call__(self, *inputs):
        print(inputs)
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 2 else outputs[0]

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

    @staticmethod
    def add(x0, x1):
        x1 = as_array(x1)
        return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

    @staticmethod
    def square(x0):
        return Square()(x0)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = list()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            # print(f"gys: {gys}, gxs: {gxs}")

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    # funcs.append(x.creator)
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

        return True

    def __repr__(self):
        if self.data is None:
            return "variable(None)"

        p = str(self.data).replace("\n", "\n{}".format(" " * 9))
        return "variable({})".format(p)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

    @staticmethod
    def mul(x0, x1):
        return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

    @staticmethod
    def neg(x):
        return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1 
        return y

    def backward(self, gy):
        return gy, -gy

    @staticmethod
    def sub(x0, x1):
        x1 = as_array(x1)
        return Sub()(x0, x1)

    @staticmethod
    def rsub(x0, x1):
        x1 = as_array(x1)
        return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.input[1].data
        gx0 = gy / x1 
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1 

    @staticmethod
    def div(x0, x1):
        x1 = as_array(x1)
        return Div()(x1, x0)

    @staticmethod
    def rdiv(x0, x1):
        x1 = as_array(x1)
        return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c 
    
    def forward(self, x):
        y = x ** self.c 
        return y 

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

    @staticmethod
    def pow(x, c):
        return Pow(c)(x)


Variable.__mul__ = Mul.mul
Variable.__add__ = Add.add
Variable.__radd__ = Add.add
Variable.__neg__ = Neg.neg
Variable.__sub__ = Sub.sub
Variable.__rsub__ = Sub.rsub
Variable.__truediv__ = Div.div 
Variable.__rtruediv__ = Div.rdiv
Variable.__pow__ = Pow.pow


if __name__ == "__main__":
    x = Variable(np.array(3.0))
    k = -x
    a = 3 - x - 2 - 4
    p = x ** 3
    print(p)
