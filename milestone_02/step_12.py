import numpy as np
from step_11 import Variable, as_array


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)

        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 2 else outputs[0]


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1

        return (y, )


if __name__ == "__main__":
    x0, x1 = Variable(np.array(2)), Variable(np.array(3))
    f = Add()
    y = f(x0, x1)

    print(y.data)