#! /usr/bin/python
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
import filter as recommend_filter 

from sklearn.decomposition import NMF
from scipy import io
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def load_dense_matrix(is_init):
    print(is_init)
    R = lil_matrix((158392, 69667))
    if is_init:
        all = pd.read_csv("../input/prepro.csv")
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

log = pd.read_csv("../input/all/log.tsv", sep="\t")
events = pd.read_csv("../input/all/filled_events.csv")
users = pd.read_csv("../input/all/users.tsv", sep="\t")
test_target = pd.read_csv("../input/all/test.tsv", sep="\t")

arg = sys.argv
is_init = True
if arg[1] == "1":
    is_init = True
else:
    is_init = False
    

R = load_dense_matrix(is_init)
print(R.shape)
model = NMF(n_components = 10, init="random", random_state=0)
W = model.fit_transform(R)
H = model.components_
w = W[0:6270,:]
h = H[:, 0:2]
# 検証用で行列を小さく小さく
result = np.dot(w, h)
user_recommend_pre_list = []
i = 1

for rating_list in result.T:
   user_recommend_pre_list.append({
       "user_id": i,
       "events": pd.DataFrame({
           "event_id": range(1, len(rating_list) + 1),
           "rating": rating_list
           })
       })
   i = i + 1
# 以下の実装は共通処理として切り出そう
# なので、このmodel.pyもmf.pyなどに直す
# 各モデルは
# {
#  "user_id":
#  "events": {
#    "event_id":,
#    "rating":
#   } 
# }
# みたいなlistを返すことを期待してfilter処理なんかを作っていこう。
# だからこのスクリプトもべたでいろいろ書いてるが最終的にはapply? fit?みたいな関数にする

join_list = recommend_filter.user_join_list(log)
test_user_recommend_list = list(filter(lambda u: u["user_id"] in test_target.user_id, user_recommend_pre_list))

user_not_join_list = recommend_filter.join_event_filter(1, test_user_recommend_list[0]["events"], join_list)
filtered_age_recommend_list = recommend_filter.age_filter(1, user_not_join_list, events, users)
ranked_recommend_list = recommend_filter.ranking(filtered_age_recommend_list).sort_values(by="ranking")
limit_ranked_recommend_list = ranked_recommend_list[ranked_recommend_list["ranking"] <= 20]
test_user_recommend_list[0]["events"] = limit_ranked_recommend_list[["event_id", "ranking"]]

print(test_user_recommend_list)
print(user_recommend_pre_list[1])
print(user_recommend_pre_list[1]["user_id"] in test_target.user_id)
print(test_target[test_target["user_id"]])
#io.savemat("result", {"result": result})
