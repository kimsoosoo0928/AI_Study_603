from sklearn import datasets
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

# iris

datasets_iris = load_iris()

x_data_iris = datasets_iris.data
y_data_iris = datasets_iris.target

# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris) # x_data 저장
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris) # y_data 저장

# boston

datasets_boston = load_boston()

x_data_boston = datasets_boston.data
y_data_boston = datasets_boston.target

np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston) 
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)

# breast cancer

datasets_cancer = load_breast_cancer()

x_data_breast_cancer = datasets_cancer.data
y_data_breast_cancer = datasets_cancer.target

np.save('./_save/_npy/k55_x_data_breast_cancer.npy', arr=x_data_breast_cancer) 
np.save('./_save/_npy/k55_y_data_breast_cancer.npy', arr=y_data_breast_cancer)

# diabetes

datasets_diabet = load_diabetes()

x_data_diabet = datasets_diabet.data
y_data_diabet = datasets_diabet.target

np.save('./_save/_npy/k55_x_data_diabet.npy', arr=x_data_diabet) 
np.save('./_save/_npy/k55_y_data_diabet.npy', arr=y_data_diabet)

# wine

datasets_wine = load_wine()

x_data_wine = datasets_wine.data
y_data_wine = datasets_wine.target

np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine) 
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)

########################################################################

# mnist

datasets_mnist = mnist

x_data_mnist = datasets_mnist.data
y_data_mnist = datasets_mnist.target

np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train_mnist) 
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train_mnist)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test_mnist) 
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test_mnist)

# fashion

datasets_fashion_mnist = fashion_mnist

x_data_fashion_mnist = datasets_fashion_mnist.data
y_data_fashion_mnist = datasets_fashion_mnist.target

np.save('./_save/_npy/k55_x_data_fashion_mnist.npy', arr=x_data_fashion_mnist) 
np.save('./_save/_npy/k55_y_data_fashion_mnist.npy', arr=y_data_fashion_mnist)

# cifar10

datasets_cifar10 = cifar10

x_data_cifar10 = datasets_cifar10.data
y_data_cifar10 = datasets_cifar10.target

np.save('./_save/_npy/k55_x_data_cifar10.npy', arr=x_data_cifar10) 
np.save('./_save/_npy/k55_y_data_cifar10.npy', arr=y_data_cifar10)

# cifar100

datasets_cifar100 = cifar100

x_data_cifar100 = datasets_cifar100.data
y_data_cifar100 = datasets_cifar100.target

np.save('./_save/_npy/k55_x_data_cifar100.npy', arr=x_data_cifar100) 
np.save('./_save/_npy/k55_y_data_cifar100.npy', arr=y_data_cifar100)