is_simple_core = True

if is_simple_core:
    from dezero.core_simple import Variable, Function
    from dezero.core_simple import Mul, Add, Neg, Sub, Div, Pow
    from dezero.core_simple import setup_variable
    from dezero.core_simple import using_config
    # from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
else:
    from dezero.core import Variable, Function

setup_variable()

