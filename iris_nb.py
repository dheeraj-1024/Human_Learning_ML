from sklearn import datasets,tree
import numpy as np

iris_dat=datasets.load_iris()
x=iris_dat.data
y=iris_dat.target
y_val=iris_dat.target_names

def prob(l):
  freq_l=[]
  for i in np.unique(l):
    count=0
    for j in range(len(l)):
      if l[j]==i:
        count+=1
    freq_l.append(count)
  sum_freq_l=np.sum(freq_l)
  prob_l=[i/sum_freq_l for i in freq_l]
  return prob_l

matrix_M=[[np.unique([x[j][k] for j in range(len(y)) if i==y[j]]) for i in np.unique(y)] for k in range(np.shape(x)[1])]
prob_M=[[prob([x[j][k] for j in range(len(y)) if i==y[j]]) for i in np.unique(y)] for k in range(np.shape(x)[1])]
prob_y=prob(y)

new_x=x[0]
ans=[]
for i in range(len(np.unique(y))):
  cond_prob=1
  for j in range(np.shape(x)[1]):
    for k in range(len(matrix_M[j][i])):
      if new_x[j]==matrix_M[j][i][k]:
        cond_prob*=prob_M[j][i][k]
        break
      elif k==len(matrix_M[j][i])-1:
        cond_prob*=0.00001 
  ans.append(prob_y[i]*cond_prob)
print([i/np.sum(ans) for i in ans])
