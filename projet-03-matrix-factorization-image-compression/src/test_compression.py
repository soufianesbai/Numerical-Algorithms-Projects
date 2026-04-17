import compression_image as comp
import numpy as np

S=np.array( [[7, 0, 0],
 [0, 5, 0],
 [0, 0, 5]])

S_new=comp.compression(S,1)
print(S_new)