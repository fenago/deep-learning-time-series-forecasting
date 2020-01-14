# load and plot daily births dataset
from pandas import read_csv
import warnings
warnings.simplefilter("ignore")
%matplotlib inline
from matplotlib import pyplot
# load
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
# summarize shape
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()