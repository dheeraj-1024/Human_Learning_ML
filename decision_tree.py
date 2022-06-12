import numpy as np
from sklearn import tree
from tensorflow.keras.datasets import mnist
(x,y),(x_val,y_val) = mnist.load_data()
x = x.reshape((60000,28*28))
x_val = x_val.reshape((10000,28*28))
model = tree.DecisionTreeClassifier(max_depth=3)
model = model.fit(x,y)
print(model.predict(x_val))
tree.plot_tree(model)
