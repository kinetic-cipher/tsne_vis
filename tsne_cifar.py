"""
  tsne_cifar.py: TSNE visualization of CIFAR10 dataset (or subset). 
 
  Author: Keith Kenemer
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.manifold import TSNE

#========== utility
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

#=========== load cifar 10 data (disregard color)
M=255 # max vlue for normalization
print('loading Cifar10 dataset')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_train_samples = x_train.shape[0]
w = x_train.shape[1]  # width
h = x_train.shape[2]  # height
x_train = np.reshape( x_train[:,:,:,0], (num_train_samples,w*h) )/M
x_train, y_train = shuffle_in_unison(x_train, y_train)
print('X:',x_train.shape)
print('Y:',y_train.shape)

#============ compute TSNE embedding
ns = 2000  # number of samples to use (FIXME: make command-ine arg)
print('starting TSNE analysis...')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_tsne= tsne.fit_transform(x_train[0:ns,:] )
print('t-SNE analysis complete. time: '+format(time.time()-time_start,'.2f')+'s'  )
print('X_tsne:',X_tsne.shape)

#=============  plot
# X_tsne: array-like, shape (n_samples, n_components)
viz_x = X_tsne[:,0]    #tsne component #1
viz_y = X_tsne[:,1]    # tsne component #2
viz_c = y_train[0:ns]  # color is the label

plt.scatter(viz_x, viz_y, c=viz_c.flatten(), cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title("TSNE on "+str(ns)+" CIFAR10 Samples")
plt.xlabel("tsne-component-1")
plt.ylabel("tsne-component-2")
plt.grid()
plt.show()


