"""
 tsne_mnist.py: TSNE visualization of MNIST dataset (or subset).

 Author: Keith Kenemer
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.manifold import TSNE

#=========== utility
# shuffle two arrays using same index permutation
def shuffle_in_unison(a, b):
   assert len(a) == len(b)
   shuffled_a = np.empty(a.shape, dtype=a.dtype)
   shuffled_b = np.empty(b.shape, dtype=b.dtype)
   permutation = np.random.permutation(len(a))
   for old_index, new_index in enumerate(permutation):
       shuffled_a[new_index] = a[old_index]
       shuffled_b[new_index] = b[old_index]
   return shuffled_a, shuffled_b

#========= load MNIST dataset
print("loading MNIST dataset")
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
N = X_train.shape[0]  # number of samples
w = X_train.shape[1]  # width of each image
h = X_train.shape[2]  # height of each image
X = np.reshape(X_train, (N,w*h)  )/255.0     # normalize
Y = Y_train           # just use training data for visualization
X,Y = shuffle_in_unison(X,Y)
print('X:',X.shape)
print('Y:',Y.shape)

#======== compute TSNE embedding
ns = 2000 # number of samples to use (FIXME: make command-line arg)
print('starting TSNE analysis...')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_tsne= tsne.fit_transform( X[0:ns,:]  )
print('t-SNE analysis complete. time: ' + format(time.time()-time_start,'.2f')+'s'  )
print('X_tsne:', X_tsne.shape)

#=========  plot
# X_tsne: array-like, shape (n_samples, n_components)
viz_x = X_tsne[:,0]
viz_y = X_tsne[:,1]
viz_c = Y[0:ns]
plt.scatter(viz_x, viz_y, c=viz_c.flatten(), cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title("TSNE on "+str(ns)+" MNIST Samples")
plt.xlabel("tsne-component-1")
plt.ylabel("tsne-component-2")
plt.grid()
plt.show()


