is_simple_core = True

if is_simple_core:
    from dezero.core_simple import Variable, Function
    from dezero.core_simple import Mul, Add, Neg, Sub, Div, Pow
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable, Function

setup_variable()

