import oct2py
import numpy as np

def eval_contours(fn,n):
 oc = oct2py.Oct2Py()
 contours = []
 for k in fn: 
  im = oc.imread("../leaves_png/"+k)
  s = oc.extract_longest_cont(im,n)
  contours.append(np.array([complex(i[0],i[1]) for i in s]))
 oc.exit() 
 return contours
