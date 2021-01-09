# 우리가 사용하는 컴퓨터는 극한을 취급할 수 없음 
# 대신 Numerical Difference(수치 미분)을 대신 사용 
# 수치 미분은 극한 대신 작은 값을 사용하여 진정한 미분을 근사하는 방식 
# 수치 미분에는 전진차분과 중앙차분이 있는데, 책에서는 중앙차분을 사용 (오차가 적음)
import numpy as np
from step_01 import Variable
from step_02_function import *
from step_03_composite_function import *

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)

    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2*eps)


# 합성 함수의 미분 
def comp_f(x):
    A = Square()
    B = Exp()
    C = Square()

    return C(B(A(x)))

'''
수치 미분에 비해서 역전파는 구현이 복잡, 그래서 버그가 섞이기 쉬움 
역전파를 제대로 구현했는지 확인하기 위해 수치미분을 사용하기도 함
이를 gradient checking(기울기 확인)이라고 함 
'''

if __name__ == "__main__":
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    x = Variable(np.array(0.5))
    dy = numerical_diff(comp_f, x)
    print(dy)
