# example of a one-step naive forecast
def naive_forecast(history, n):
	return history[-n]

# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# test naive forecast
for i in range(1, len(data)+1):
	print(naive_forecast(data, i))