# example of an average forecast for seasonal data
from numpy import mean
from numpy import median

# one-step average forecast
def average_forecast(history, config):
	n, offset, avg_type = config
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# mean of last n values
	if avg_type is 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# define dataset
data = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
print(data)
# test naive forecast
for i in [1, 2, 3]:
	print(average_forecast(data, (i, 3, 'mean')))