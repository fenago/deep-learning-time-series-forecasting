# example of dropping the time dimension from the dataset
from numpy import array

# define the dataset
data = list()
n = 5000
for i in range(n):
	data.append([i+1, (i+1)*10])
data = array(data)
# drop time
data = data[:, 1]
print(data.shape)