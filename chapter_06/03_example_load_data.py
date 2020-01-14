# example of defining a dataset
from numpy import array

# define the dataset
data = list()
n = 5000
for i in range(n):
	data.append([i+1, (i+1)*10])
data = array(data)
print(data[:5, :])
print(data.shape)