# example of an average forecast
from numpy import mean
from numpy import median

# one-step average forecast
def average_forecast(history, config):
	n, avg_type = config
	# mean of last n values
	if avg_type is 'mean':
		return mean(history[-n:])
	# median of last n values
	return median(history[-n:])

# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# test naive forecast
for i in range(1, len(data)+1):
	print(average_forecast(data, (i, 'mean')))