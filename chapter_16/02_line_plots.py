# line plots for power usage dataset
from pandas import read_csv
import warnings
warnings.simplefilter("ignore")
%matplotlib inline
from matplotlib import pyplot
# load the new file
dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# line plot for each variable
pyplot.figure()
for i in range(len(dataset.columns)):
	# create subplot
	pyplot.subplot(len(dataset.columns), 1, i+1)
	# get variable name
	name = dataset.columns[i]
	# plot data
	pyplot.plot(dataset[name])
	# set title
	pyplot.title(name, y=0)
	# turn off ticks to remove clutter
	pyplot.yticks([])
	pyplot.xticks([])
pyplot.show()