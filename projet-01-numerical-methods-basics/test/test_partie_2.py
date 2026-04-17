import pytest
import sys
sys.path.append('../src')  
import math
from partie_2 import log2_approximatif, ln_CORDIC, exp_CORDIC, arctan_CORDIC, tan_CORDIC

def test_log2_approx():
    for p in range(1, 15):
        err, log2_approx_result = log2_approximatif(p)
        assert(abs(log2_approx_result - math.log(2)) < 2**(-p))

def test_ln_CORDIC():
    try:
        ln_CORDIC(0)  
    except ValueError:
        pass  

    assert(abs(ln_CORDIC(1e10) - math.log(1e10)) < 1e-4)
    assert(abs(ln_CORDIC(10) - math.log(10)) < 1e-4)
    assert(abs(ln_CORDIC(2) - math.log(2)) < 1e-4)
    assert(abs(ln_CORDIC(1) - math.log(1)) < 1e-4) 
    assert(abs(ln_CORDIC(0.9) - math.log(0.9)) < 1e-2)

def test_exp_CORDIC():
    assert(abs(exp_CORDIC(1) - math.exp(1)) < 1e-4)
    assert(abs(exp_CORDIC(0.5) - math.exp(0.5)) < 1e-4)
    assert(abs(exp_CORDIC(0) - math.exp(0)) < 1e-4)
    assert(abs(exp_CORDIC(2.7) - math.exp(2.7)) < 1e-4)
    assert(abs(exp_CORDIC(3) - math.exp(3)) < 1e-4)
    assert(abs(exp_CORDIC(5) - math.exp(5)) < 1e-4)
    assert(abs(exp_CORDIC(20) - math.exp(20)) < 1e-4)

def test_arctan_CORDIC():
    assert(abs(arctan_CORDIC(0.1) - math.atan(0.1)) < 1e-4)
    assert(abs(arctan_CORDIC(1) - math.atan(1)) < 1e-4)
    assert(abs(arctan_CORDIC(-1) - math.atan(-1)) < 1e-4)
    assert(abs(arctan_CORDIC(0) - math.atan(0)) < 1e-4)
    assert(abs(arctan_CORDIC(10) - math.atan(10)) < 1e-4)

def test_tan_CORDIC():
    assert(abs(tan_CORDIC(0.1) - math.tan(0.1)) < 1e-4)
    assert(abs(tan_CORDIC(0.5) - math.tan(0.5)) < 1e-4)
    assert(abs(tan_CORDIC(1) - math.tan(1)) < 1e-4)
    assert(abs(tan_CORDIC(2) - math.tan(2)) < 1e-4)
    assert(abs(tan_CORDIC(5) - math.tan(5)) < 1e-4)    


if __name__ == "__main__":
    pytest.main()
