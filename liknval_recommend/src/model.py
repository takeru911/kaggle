#! /usr/bin/python
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import scipy.sparse as sp
from scipy import io
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def load_dense_matrix(is_init, all):
    print(is_init)
    R = lil_matrix((158392, 69667))
    if is_init:
        for index, d in all.iterrows():
          if index % 1000 == 0:
            print("num iter: {}".format(index))

          event_id = d.event_id -1
          user_id = d.user_id -1
          R[event_id, user_id] = d.action_type
        io.savemat("../input/model/R_matrix", {"R": R})
        return R
    else:
        R = io.loadmat("../input/model/R_matrix")["R"]
        return R

arg = sys.argv
is_init = True
if arg[1] == "1":
    is_init = True
else:
    is_init = False
    
all = pd.read_csv("../input/prepro.csv")
R = load_dense_matrix(is_init, all)
print(R.shape)
model = NMF(n_components = 10, init="random", random_state=0)
W = model.fit_transform(R)
H = model.components_
print(W)
#result = np.dot(W, H)
# expect 3
#print(result[125664, 11045])
# expect 1
#print(result[125664, 0])
#io.savemat("result", {"result": result})
