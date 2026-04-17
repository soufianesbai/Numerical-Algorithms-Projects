import math 
import numpy as np
from partie_1 import rp_sum,rp

def log2_approximatif(p):
    log2_reel = math.log(2)
    log2_approx = 0 
    terme = 0 
    n = 1   

    while True:
        terme = ((-1)**(n + 1)) / n
        log2_approx = rp_sum(log2_approx, rp(terme,p), p)
        err = abs((log2_approx - log2_reel) / log2_approx)
        if (err < 2**(-p)):
            break
        n += 1
        
    return err, rp(log2_approx,p)


L = [math.log(1 + 10**-k) for k in range(7)]
A = [math.atan(10**-k) for k in range(5)]

def ln_CORDIC(x):
    if x <= 0:
        raise ValueError("ln(x) non défini pour x <= 0")
        
    k = 0
    y = 0.0
    p = 1.0  
    
    while k <= 6:
        while x >= p + p * 10**-k:
            y += L[k]  
            p += p * 10**-k  
        
        k += 1  
    
    return y + (x / p - 1)

def exp_CORDIC(x):
    if x >= math.log(10):
        return exp_CORDIC(x - math.log(10)) * 10
    
    k = 0
    y = 1.0 
    
    while k <= 6:
        while x >= L[k]:
            x -= L[k]  
            y += y * 10**-k  
        
        k += 1
    
    return y + y * x  

def arctan_CORDIC(x):
    if  x > 1:
        return math.pi/2 - arctan_CORDIC(1/x)
    if x < 0:
        return -arctan_CORDIC(-x)
    
    k = 0
    y = 1.0
    r = 0.0  
    
    while k < 5:
        while x < y * 10**-k:
            k += 1 

        if k >= 5:
            break

        xp = x - y * 10**-k
        y = y + x * 10**-k
        x = xp
        r += A[k]     
         
    return r + x / y  


def tan_CORDIC(x):
    x = x % math.pi
    if x > math.pi/2:
        return -tan_CORDIC(math.pi - x)
    if x > math.pi/4:
        return 1 / tan_CORDIC(math.pi/2 - x)

    k = 0
    n = 0.0  
    d = 1.0 
    
    while k <= 4:
        while x >= A[k]:
            x -= A[k]  
            np = n + d * 10**-k  
            d = d - n * 10**-k   
            n = np
        
        k += 1
    
    return (n + x * d) / (d - x * n)




