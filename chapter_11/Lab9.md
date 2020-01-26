<img align="right" src="../logo-small.png">


### How to Develop Simple Methods for Univariate Forecasting (Part 2)

Simple forecasting methods include naively using the last observation as the prediction or an
average of prior observations. It is important to evaluate the performance of simple forecasting
methods on univariate time series forecasting problems before using more sophisticated methods
as their performance provides a lower-bound and point of comparison that can be used to
determine of a model has skill or not for a given problem. Although simple, methods such as
the naive and average forecast strategies can be tuned to a specific problem in terms of the
choice of which prior observation to persist or how many prior observations to average. Often,
tuning the hyperparameters of these simple strategies can provide a more robust and defensible
lower bound on model performance, as well as surprising results that may inform the choice and
configuration of more sophisticated methods.
In this tutorial, you will discover how to develop a framework from scratch for grid searching
simple naive and averaging strategies for time series forecasting with univariate data. After
completing this tutorial, you will know:

- How to develop a framework for grid searching simple models from scratch using walk-
forward validation.

- How to grid search simple model hyperparameters for daily time series
data for births.

- How to grid search simple model hyperparameters for monthly time series data for shampoo
sales, car sales, and temperature.

Let' s get started.

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-time-series-forecasting` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab9_Simple_Methods_Univariate_Forecasting_Part2`


### Case Study 2: Trend

The monthly shampoo salesdataset summarizes the monthly sales of shampoo
over a three-year period. You can download the dataset directly from here:

- monthly-shampoo-sales.csv^4

Save the file with the filename monthly-shampoo-sales.csv in your current working directory. We can load this dataset as a Pandas Series using the function read csv() and
summarize the shape of the dataset.


```
series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)

print(series.shape)

```

We can then create a line plot of the series and inspect it for systematic structures like
trends and seasonality.

```
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()

```

The complete example is listed below.

```
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)

print(series.shape)
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()

```

##### Run Notebook
Click notebook `07_load_plot_monthly_shampoo.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first summarizes the shape of the loaded dataset. The dataset has
three years, or 36 observations. We will use the first 24 for training
and the remaining 12 as the test set.

```
(36, 1)

```

A line plot of the series is also created. We can see that there is an obvious trend and no
obvious seasonality.


![](./images/207-9.png)

We can now grid search naive models for the dataset. The complete
example grid searching
the shampoo sales univariate time series forecasting problem is listed
below.

```
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv

def simple_forecast(history, config):
n, offset, avg_type = config

if avg_type == 'persist':
return history[-n]

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
# check if we can average
if len(values) < 2:
raise Exception('Cannot calculate average')
# mean of last n values
if avg_type == 'mean':
return mean(values)
# median of last n values
return median(values)

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = simple_forecast(history, cfg)

predictions.append(yhat)

history.append(test[i])

error = measure_rmse(test, predictions)
return error

def score_model(data, n_test, cfg, debug=False):
result = None

key = str(cfg)

if debug:
result = walk_forward_validation(data, n_test, cfg)
else:

try:


# never show warnings when grid searching, too noisy
with catch_warnings():
filterwarnings("ignore")
result = walk_forward_validation(data, n_test, cfg)
except:
error = None
# check for an interesting result
if result is not None:
print('> Model[%s] %.3f' % (key, result))
return (key, result)

def grid_search(data, cfg_list, n_test, parallel=True):
scores = None
if parallel:

executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in
cfg_list)
scores = executor(tasks)
else:
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]

scores = [r for r in scores if r[1] != None]

scores.sort(key=lambda tup: tup[1])
return scores

def simple_configs(max_length, offsets=[1]):
configs = list()
for i in range(1, max_length+1):
for o in offsets:
for t in ['persist', 'mean', 'median']:
cfg = [i, o, t]
configs.append(cfg)
return configs

if **name** =='**main**':
series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)
data = series.values
n_test = 12
max_length = len(data) - n_test
cfg_list = simple_configs(max_length)
scores = grid_search(data, cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```


##### Run Notebook
Click notebook `08_grid_search_shampoo_sales.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prints the configurations and the RMSE are printed
as the models are
evaluated. The top three model configurations and their error are reported at the end of the
run.


```
...
> Model[[23, 1,'mean']] 209.782
> Model[[23, 1,'median']] 221.863
> Model[[24, 1,'persist']] 305.635
> Model[[24, 1,'mean']] 213.466
> Model[[24, 1,'median']] 226.061
done

[2, 1,'persist'] 95.69454007413378
[2, 1,'mean'] 96.01140340258198
[2, 1,'median'] 96.01140340258198

```

We can see that the best result was an RMSE of about 95.69 sales with the following
configuration:

- Strategy: Persist

- n: 2

This is surprising as the trend structure of the data would suggest that
persisting the previous

value (-1) would be the best approach, not persisting the second last
value.

#### Case Study 3: Seasonality

The monthly mean temperatures dataset summarizes the monthly average air temperatures in
Nottingham Castle, England from 1920 to 1939 in degrees Fahrenheit. You can download the
dataset directly from here:

- monthly-mean-temp.csv 5

Save the file with the filename monthly-mean-temp.csv in your current working directory.
We can load this dataset as a Pandas Series using the function read csv() and summarize
the shape of the dataset.

```
# load
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
# summarize shape
print(series.shape)

```

We can then create a line plot of the series and inspect it for systematic structures like
trends and seasonality.

```
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()

```
The complete example is listed below.

```
# load and plot monthly mean temp dataset
from pandas import read_csv
from matplotlib import pyplot
# load
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
# summarize shape
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()

```


##### Run Notebook
Click notebook `09_load_plot_monthly_mean_temp.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first summarizes the shape of the loaded dataset.
The dataset has 20
years, or 240 observations.


```
(240, 1)

```

A line plot of the series is also created. We can see that there is no obvious trend and an
obvious seasonality structure.


![](./images/212-10.png)

We will trim the dataset to the last five years of data (60 observations) in order to speed up
the model evaluation process and use the last year or 12 observations for the test set.

```
data = data[-(5*12):]

```

The period of the seasonal component is about one year, or 12 observations. We will use
this as the seasonal period in the call to the simple configs() function when preparing the
model configurations.

```
cfg_list = simple_configs(seasonal=[0, 12])

```

We can now grid search naive models for the dataset. The complete example grid searching
the monthly mean temperature time series forecasting problem is listed below.

```
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count


from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv

def simple_forecast(history, config):
n, offset, avg_type = config

if avg_type == 'persist':
return history[-n]

values = list()
if offset == 1:
values = history[-n:]
else:

if n*offset > len(history):
raise Exception('Config beyond end of data: %d %d' % (n,offset))

for i in range(1, n+1):
ix = i * offset
values.append(history[-ix])

if len(values) < 2:
raise Exception('Cannot calculate average')

if avg_type == 'mean':
return mean(values)

return median(values)

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = simple_forecast(history, cfg)

predictions.append(yhat)


history.append(test[i])
# estimate prediction error
error = measure_rmse(test, predictions)
return error

def score_model(data, n_test, cfg, debug=False):
result = None

key = str(cfg)

if debug:
result = walk_forward_validation(data, n_test, cfg)
else:

try:

with catch_warnings():
filterwarnings("ignore")
result = walk_forward_validation(data, n_test, cfg)
except:
error = None

if result is not None:
print('> Model[%s] %.3f' % (key, result))
return (key, result)

def grid_search(data, cfg_list, n_test, parallel=True):
scores = None
if parallel:

executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in
cfg_list)
scores = executor(tasks)
else:
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]

scores = [r for r in scores if r[1] != None]

scores.sort(key=lambda tup: tup[1])
return scores

def simple_configs(max_length, offsets=[1]):
configs = list()
for i in range(1, max_length+1):
for o in offsets:
for t in ['persist', 'mean', 'median']:
cfg = [i, o, t]
configs.append(cfg)
return configs

if **name** =='**main**':
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)


data = series.values
# data split
n_test = 12
# model configs
max_length = len(data) - n_test
cfg_list = simple_configs(max_length, offsets=[1,12])
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
print(cfg, error)

```


##### Run Notebook
Click notebook `10_grid_search_mean_temp.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prints the model configurations and the RMSE are printed as the
models are evaluated. The top three model configurations and their error are reported at the
end of the run.

```
> Model[[227, 12,'persist']] 5.365
> Model[[228, 1,'persist']] 2.818
> Model[[228, 1,'mean']] 8.258
> Model[[228, 1,'median']] 8.361
> Model[[228, 12,'persist']] 2.818
done

[4, 12,'mean'] 1.5015616870445234
[8, 12,'mean'] 1.5794579766489512
[13, 12,'mean'] 1.586186052546763

```

We can see that the best result was an RMSE of about 1.50 degrees with the following
configuration:
- Strategy: Average
- n: 4
- offset: 12
- function: mean()

This finding is not too surprising. Given the seasonal structure of the data, we would expect
a function of the last few observations at prior points in the yearly cycle to be effective.

### Case Study 4: Trend and Seasonality

The monthly car salesdataset summarizes the monthly car sales in Quebec,
Canada between
1960 and 1968. You can download the dataset directly from here:

- monthly-car-sales.csv

Save the file with the filename monthly-car-sales.csv in your current working directory.
We can load this dataset as a Pandas Series using the function read csv() and summarize
the shape of the dataset.

```
# load
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# summarize shape
print(series.shape)

```
We can then create a line plot of the series and inspect it for systematic structures like
trends and seasonality.

```
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()

```
The complete example is listed below.

```
# load and plot monthly car sales dataset
from pandas import read_csv
from matplotlib import pyplot
# load
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# summarize shape
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()

```


##### Run Notebook
Click notebook `11_load_plot_monthly_car_sales.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first summarizes the shape of the loaded dataset. The dataset has 9
years, or 108 observations. We will use the last year or 12 observations
as the test set.

```
(108, 1)

```

A line plot of the series is also created. We can see that there is an obvious trend and
seasonal components.


![](./images/217-11.png)

The period of the seasonal component could be six months or 12 months.
We will try both
as the seasonal period in the call to thesimpleconfigs() function when
preparing the model
configurations.

```
cfg_list = simple_configs(seasonal=[0,6,12])

```

We can now grid search naive models for the dataset. The complete
example grid searching
the monthly car sales time series forecasting problem is listed below.

```
from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from pandas import read_csv


def simple_forecast(history, config):
n, offset, avg_type = config

if avg_type == 'persist':
return history[-n]

values = list()
if offset == 1:
values = history[-n:]
else:

if n*offset > len(history):
raise Exception('Config beyond end of data: %d %d' % (n,offset))

for i in range(1, n+1):
ix = i * offset
values.append(history[-ix])

if len(values) < 2:
raise Exception('Cannot calculate average')

if avg_type == 'mean':
return mean(values)

return median(values)

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = simple_forecast(history, cfg)

predictions.append(yhat)

history.append(test[i])

error = measure_rmse(test, predictions)
return error

def score_model(data, n_test, cfg, debug=False):


result = None
# convert config to a key
key = str(cfg)
# show all warnings and fail on exception if debugging
if debug:
result = walk_forward_validation(data, n_test, cfg)
else:
# one failure during model validation suggests an unstable config
try:
# never show warnings when grid searching, too noisy
with catch_warnings():
filterwarnings("ignore")
result = walk_forward_validation(data, n_test, cfg)
except:
error = None
# check for an interesting result
if result is not None:
print('> Model[%s] %.3f' % (key, result))
return (key, result)

def grid_search(data, cfg_list, n_test, parallel=True):
scores = None
if parallel:

executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in
cfg_list)
scores = executor(tasks)
else:
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]

scores = [r for r in scores if r[1] != None]

scores.sort(key=lambda tup: tup[1])
return scores

def simple_configs(max_length, offsets=[1]):
configs = list()
for i in range(1, max_length+1):
for o in offsets:
for t in ['persist', 'mean', 'median']:
cfg = [i, o, t]
configs.append(cfg)
return configs

if **name** =='**main**':
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
data = series.values
n_test = 12
max_length = len(data) - n_test
cfg_list = simple_configs(max_length, offsets=[1,12])

scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
print(cfg, error)

```

##### Run Notebook
Click notebook `12_grid_search_car_sales.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prints the model configurations and the RMSE are printed as the
models are evaluated. The top three model configurations and their error are reported at the
end of the run.

```
> Model[[79, 1,'median']] 5124.113
> Model[[91, 12,'persist']] 9580.149
> Model[[79, 12,'persist']] 8641.529
> Model[[92, 1,'persist']] 9830.921
> Model[[92, 1,'mean']] 5148.126
done

[3, 12,'median'] 1841.1559321976688
[3, 12,'mean'] 2115.198495632485
[4, 12,'median'] 2184.37708988932

```

We can see that the best result was an RMSE of about 1841.15 sales with the following
configuration:

- Strategy: Average

- n: 3

- offset: 12

- function: median()

It is not surprising that the chosen model is a function of the last few observations at the
same point in prior cycles, although the use of the median instead of the mean may not have
been immediately obvious and the results were much better than the mean.

## Exercises

This section lists some ideas for extending the tutorial that you may
wish to explore.

- Plot Forecast. Update the framework to re-fit a model with the best configuration and
forecast the entire test dataset, then plot the forecast compared to the actual observations
in the test set.

- Drift Method. Implement the drift method for simple forecasts and compare the results
to the average and naive methods.


- Another Dataset. Apply the developed framework to an additional univariate time
series problem (e.g. from the Time Series Dataset Library).


### Further Reading

This section provides more resources on the topic if you are looking to
go deeper.

#### Datasets

- Time Series Dataset Library, DataMarket.
https://datamarket.com/data/list/?q=provider:tsdl

- Daily Female Births Dataset, DataMarket.
https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959

- Monthly Shampoo Sales Dataset, DataMarket.
https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period

- Monthly Mean Temperature Dataset, DataMarket.
https://datamarket.com/data/set/22li/mean-monthly-air-temperature-deg-f-nottingham-castle-1920-1939

- Monthly Car Sales Dataset, DataMarket.
https://datamarket.com/data/set/22n4/monthly-car-sales-in-quebec-1960-1968

#### APIs

- numpy.meanAPI.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html

- numpy.medianAPI.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

- sklearn.metrics.meansquarederrorAPI.
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_
error.html

- pandas.readcsvAPI.
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

- Joblib: running Python functions as pipeline jobs.
https://pythonhosted.org/joblib/

#### Articles

- Forecasting, Wikipedia.
https://en.wikipedia.org/wiki/Forecasting


#### Summary

In this tutorial, you discovered how to develop a framework from scratch for grid searching
simple naive and averaging strategies for time series forecasting with univariate data. Specifically,

you learned:

- How to develop a framework for grid searching simple models from scratch using walk-
forward validation.

- How to grid search simple model hyperparameters for daily time series
data for births.

- How to grid search simple model hyperparameters for monthly time series data for shampoo
sales, car sales, and temperature.

#### Next

In the next lab, you will discover how to develop exponential smoothing models for univariate
time series forecasting problems.

