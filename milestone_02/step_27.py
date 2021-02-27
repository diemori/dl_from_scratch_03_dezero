import numpy as np

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero import Variable
from dezero import Sin
from dezero.utils import get_dot_graph, plot_dot_graph


x = Variable(np.array(np.pi/4))
y = Sin.sin(x)
y.backward()

print(y.data)
print(x.grad)

