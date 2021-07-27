import numpy as np

x_data = np.load('./_save/_npy/k55_x_data_wine_mnist.npy')
y_data = np.load('./_save/_npy/k55_y_data_wine_mnist.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)