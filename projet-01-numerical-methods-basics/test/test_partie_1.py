import pytest
import sys
sys.path.append('../src')  
from partie_1 import rp, rp_sum, rp_multiply, sum_error, multiply_error

def test_rp():
    assert(rp(3.141592658, 4) == 3.142)
    assert(rp(3.141592658, 6) == 3.14159)
    assert(rp(10507.1823, 4) == 10510)
    assert(rp(10507.1823, 6) == 10507.2)
    assert(rp(0.0001857563, 4) == 0.0001858)
    assert(rp(0.0001857563, 6) == 0.000185756)

def test_rp_sum():
    assert(rp_sum(105.354, 384.34, 4) == 489.7)
    assert(rp_sum(0.00345, 0.980, 3) == 0.983)
    assert(rp_sum(3.654, 6.23, 5) == 9.8840)

def test_rp_multiply():
    assert(rp_multiply(0.00345, 0.234, 5) == 0.00080730)
    assert(rp_multiply(4.9584, 2435.34, 6) == 12075.4)
    assert(rp_multiply(5424, 34, 4) == 184400)
    assert(rp_multiply(0.0000012343, 132443, 4) == 0.1634)
    assert(rp_multiply(0.000076343, 9324.43, 4) == 0.7118)    

def test_sum_error():
    assert(sum_error(145341, 433413, 3) == 0.0013027987711532016)
    assert(sum_error(12473427921844792, 34143453749583, 4) == 0.0006053433849795798)
    assert(sum_error(3, 23, 5) == 0)

def test_multiply_error():
    assert(multiply_error(88234, 573493, 6) == 3.683284098705358e-07)
    assert(multiply_error(4, 93, 3) == 0)
    assert(multiply_error(3425452353534534534, 2353453432453443459813342, 6) == 1.5628491863858925e-06)

if __name__ == "__main__":
    pytest.main()