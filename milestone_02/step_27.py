import math
import numpy as np

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero import Variable
from dezero import Sin
from dezero.utils import get_dot_graph, plot_dot_graph

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(10000):
        c = (-1) ** i / math.factorial(2*i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t

        if abs(t.data) < threshold:
            break

    return y

x = Variable(np.array(np.pi/4))
y = my_sin(x, threshold=1e-20)
y.backward()


plot_dot_graph(y, to_file="step27.png")