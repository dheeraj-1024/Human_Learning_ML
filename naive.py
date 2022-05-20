#                           **************************************************************
#                           * Naive-Bayes classifier to classify 0 and 8 from mnist data *
#                           **************************************************************

import numpy  as np
%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x,y),(x_val,y_val)=mnist.load_data()
a,b=[],[]
for i in range(len(x)):
  if y[i]==0 or y[i]==8:
    a.append(x[i])
    b.append(y[i])

a_flat=np.reshape(a,(np.shape(a)[0],28*28))
a_fin=a_flat/np.max(a_flat)

#preparing validation data
a_val,b_val=[],[]
for i in range(len(x_val)):
  if y_val[i]==0 or y_val[i]==8:
    a_val.append(x_val[i])
    b_val.append(y_val[i])

a_val_flat=np.reshape(a_val,(np.shape(a_val)[0],28*28))
a_val_fin=a_val_flat/np.max(a_val_flat)

import numpy as np
def naive_bayes(x,y,new_x,accu=0.01):
 count_1,count_2=0,0
 list_1,list_2=[],[]
 for i in range(len(y)):
  if y[i]==0:
   count_1+=1
   list_1.append(i)
 for j in range(len(y)):
  if y[j]==8:
   count_2+=1
   list_2.append(j)
 prob_y_zero=count_1/len(y)
 prob_y_one=count_2/len(y)
 
 prob_1=1
 for iter_1 in range(np.shape(x)[1]):
  count=0
  for iter_2 in list_1:
   if abs(x[iter_2][iter_1]-new_x[iter_1])==accu:
    count+=1
  const_1=(count/count_1)
  if const_1==0.0:
   continue
  else:
   prob_1=prob_1+const_1

 prob_2=1
 for iter_1 in range(np.shape(x)[1]):
  count_3=0
  for iter_2 in list_2:
   if abs(x[iter_2][iter_1]-new_x[iter_1])==accu:
    count_3+=1
  const_2=(count_3/count_2)
  if const_2==0.0:
   continue
  else:
   prob_2=prob_2+const_2

 ans=np.array([prob_1*prob_y_zero,prob_2*prob_y_one])
 ans_1=ans/np.sum(ans)
 if ans_1[0]>ans_1[1]:
   return 0
 else:
   return 8

for i in range(20):
 print("test value is ",b_val[i],"predicted value is ",naive_bayes(a_fin,b,a_val_fin[i],0.0))
