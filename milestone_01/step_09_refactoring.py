import numpy as np
from step_08_iteration import Square, Exp, Variable

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def run_new_method():
    x = Variable(np.array(0.5))
    # a = square(x)
    # b = exp(a)
    # y = square(b)
    y = square(exp(square(x)))

    # y.grad = np.array(1.0)
    y.backward()

    print(x.grad)

run_new_method()
