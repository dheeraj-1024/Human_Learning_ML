from sklearn import datasets,tree
import numpy as np

iris_dat=datasets.load_iris()
x_train=iris_dat.data
y_train=iris_dat.target
#y_val=iris_dat.target_names

class naive_bayes:
  def prob(self,l):
    self.l=l
    self.freq_l=[]
    for i in np.unique(self.l):
      self.count=0
      for j in range(len(self.l)):
        if self.l[j]==i:
          self.count+=1
      self.freq_l.append(self.count)
    self.sum_freq_l=np.sum(self.freq_l)
    self.prob_l=[i/self.sum_freq_l for i in self.freq_l]
    return self.prob_l

  def fit(self,x,y):
    self.x=x
    self.y=y
    self.matrix_M=[[np.unique([self.x[j][k] for j in range(len(self.y)) if i==self.y[j]]) for i in np.unique(self.y)] for k in range(np.shape(self.x)[1])]
    self.prob_M=[[self.prob([self.x[j][k] for j in range(len(self.y)) if i==self.y[j]]) for i in np.unique(self.y)] for k in range(np.shape(self.x)[1])]
    self.prob_y=self.prob(self.y)
    return self.matrix_M,self.prob_M,self.prob_y

  def predict(self,matrix_N,new_x):
    self.matrix_N=matrix_N 
    self.new_x=new_x
    self.ans=[]
    for i in range(len(np.unique(self.y))):
      self.cond_prob=1
      for j in range(np.shape(self.x)[1]):
        for k in range(len(self.matrix_N[0][j][i])):
          if self.new_x[j]==self.matrix_N[0][j][i][k]:
            self.cond_prob*=self.matrix_N[1][j][i][k]
            break
          elif k==len(self.matrix_N[0][j][i])-1:
            self.cond_prob*=0.00001 
      self.ans.append(self.matrix_N[2][i]*self.cond_prob)
    return np.unique(self.y)[np.argmax(self.ans)]

model=naive_bayes()
model_train=model.fit(x_train,y_train)
print(model.predict(model_train,[7.,3.,6,2]))
