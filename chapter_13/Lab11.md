<img align="right" src="../logo-small.png">


### How to Develop SARIMA Models for Univariate Forecasting

The Seasonal Autoregressive Integrated Moving Average, or SARIMA, model
is an approach
for modeling univariate time series data that may contain trend and seasonal components. It
is an effective approach for time series forecasting, although it requires careful analysis and
domain expertise in order to configure the seven or more model hyperparameters. An alternative
approach to configuring the model that makes use of fast and parallel modern hardware is to
grid search a suite of hyperparameter configurations in order to discover what works best. Often,
this process can reveal non-intuitive model configurations that result in lower forecast error
than those configurations specified through careful analysis.
In this tutorial, you will discover how to develop a framework for grid searching all of the
SARIMA model hyperparameters for univariate time series forecasting. After completing this
tutorial, you will know:

- How to develop a framework for grid searching SARIMA models from scratch using
walk-forward validation.

- How to grid search SARIMA model hyperparameters for daily time series
data for births.

- How to grid search SARIMA model hyperparameters for monthly time series data for
shampoo sales, car sales, and temperature.

Let’s get started.

### Tutorial Overview

This tutorial is divided into five parts; they are:

1.  Develop a Grid Search Framework
2.  Case Study 1: No Trend or Seasonality
3.  Case Study 2: Trend
4.  Case Study 3: Seasonality
5.  Case Study 4: Trend and Seasonality

### Develop a Grid Search Framework

In this section, we will develop a framework for grid searching SARIMA model hyperparameters
for a given univariate time series forecasting problem. For more information on SARIMA for
time series forecasting, see Chapter 5. We will use the implementation of SARIMA provided by
the Statsmodels library. This model has hyperparameters that control the nature of the model
performed for the series, trend and seasonality, specifically:

- order: A tuplep,d, andqparameters for the modeling of the trend.

- seasonalorder: A tuple of P,D,Q, andmparameters for the modeling the
seasonality

- trend: A parameter for controlling a model of the deterministic trend as one of‘n’,‘c’,
‘t’, and‘ct’for no trend, constant, linear, and constant with linear trend, respectively.

If you know enough about your problem to specify one or more of these parameters, then

you should specify them. If not, you can try grid searching these
parameters. We can start-off

by defining a function that will fit a model with a given configuration and make a one-step
forecast. Thesarimaforecast()below implements this behavior.
The function takes an array or list of contiguous prior observations and a list of configuration
parameters used to configure the model, specifically two tuples and a string for the trend
order, seasonal order trend, and parameter. We also try to make the model robust by relaxing
constraints, such as that the data must be stationary and that the MA transform be invertible.

```
# one-step sarima forecast
def sarima_forecast(history, config):
order, sorder, trend = config
# define model
model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
enforce_stationarity=False, enforce_invertibility=False)
# fit model
model_fit = model.fit(disp=False)
# make one step forecast
yhat = model_fit.predict(len(history), len(history))
return yhat[0]

```
In this tutorial, we will use the grid searching framework developed in Chapter 11 for tuning
and evaluating naive forecasting methods. One important modification to the framework is the
function used to perform the walk-forward validation of the model namedwalkforwardvalidation().

This function must be updated to call the function for making an SARIMA
forecast. The
updated version of the function is listed below.

```
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
predictions = list()
# split dataset
train, test = train_test_split(data, n_test)
# seed history with training dataset
history = [x for x in train]
# step over each time step in the test set
for i in range(len(test)):


# fit model and make forecast for history
yhat = sarima_forecast(history, cfg)
# store forecast in list of predictions
predictions.append(yhat)
# add actual observation to history for the next loop
history.append(test[i])
# estimate prediction error
error = measure_rmse(test, predictions)
return error

```

We’re nearly done. The only thing left to do is to define a list of model configurations to
try for a dataset. We can define this generically. The only parameter we may want to specify
is the periodicity of the seasonal component in the series, if one exists. By default, we will
assume no seasonal component. Thesarimaconfigs() function below will create a list of
model configurations to evaluate.
The configurations assume each of the AR, MA, and I components for trend and seasonality
are low order, e.g. off (0) or in[1,2]. You may want to extend these ranges if you believe the
order may be higher. An optional list of seasonal periods can be specified, and you could even
change the function to specify other elements that you may know about your time series. In
theory, there are 1,296 possible model configurations to evaluate, but in practice, many will not
be valid and will result in an error that we will trap and ignore.

```
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
models = list()
# define config lists
p_params = [0, 1, 2]
d_params = [0, 1]
q_params = [0, 1, 2]
t_params = ['n','c','t','ct']
P_params = [0, 1, 2]
D_params = [0, 1]
Q_params = [0, 1, 2]
m_params = seasonal
# create config instances
for p in p_params:
for d in d_params:
for q in q_params:
for t in t_params:
for P in P_params:
for D in D_params:
for Q in Q_params:
for m in m_params:
cfg = [(p,d,q), (P,D,Q,m), t]
models.append(cfg)
return models

```

We now have a framework for grid searching SARIMA model hyperparameters
via one-step
walk-forward validation. It is generic and will work for any in-memory
univariate time series
provided as a list or NumPy array. We can make sure all the pieces work
together by testing it
on a contrived 10-step dataset. The complete example is listed below.

```

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def sarima_forecast(history, config):
order, sorder, trend = config
model = SARIMAX(history, order=order, seasonal_order=sorder,
trend=trend,
enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(history), len(history))
return yhat[0]

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = sarima_forecast(history, cfg)

predictions.append(yhat)

history.append(test[i])

error = measure_rmse(test, predictions)
return error

def score_model(data, n_test, cfg, debug=False):
result = None


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

def sarima_configs(seasonal=[0]):
models = list()

p_params = [0, 1, 2]
d_params = [0, 1]
q_params = [0, 1, 2]
t_params = ['n','c','t','ct']
P_params = [0, 1, 2]
D_params = [0, 1]
Q_params = [0, 1, 2]
m_params = seasonal

for p in p_params:
for d in d_params:
for q in q_params:
for t in t_params:
for P in P_params:
for D in D_params:
for Q in Q_params:
for m in m_params:
cfg = [(p,d,q), (P,D,Q,m), t]


models.append(cfg)
return models

if __name__ =='__main__':
# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# data split
n_test = 4
# model configs
cfg_list = sarima_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
print(cfg, error)

```
Running the example first prints the contrived time series dataset. Next, the model
configurations and their errors are reported as they are evaluated, truncated below for brevity.
Finally, the configurations and the error for the top three configurations are reported. We can
see that many models achieve perfect performance on this simple linearly increasing contrived
time series problem.

```

[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

> Model[[(2, 0, 0), (2, 0, 0, 0),'ct']] 0.001
> Model[[(2, 0, 0), (2, 0, 1, 0),'ct']] 0.000
> Model[[(2, 0, 1), (0, 0, 0, 0),'n']] 0.000
> Model[[(2, 0, 1), (0, 0, 1, 0),'n']] 0.000
done

[(2, 1, 0), (1, 0, 0, 0),'n'] 0.0
[(2, 1, 0), (2, 0, 0, 0),'n'] 0.0
[(2, 1, 1), (1, 0, 1, 0),'n'] 0.0

```

Now that we have a robust framework for grid searching SARIMA model hyperparameters,
let’s test it out on a suite of standard univariate time series datasets. The datasets were chosen
for demonstration purposes; I am not suggesting that a SARIMA model is the best approach
for each dataset; perhaps an ETS or something else would be more appropriate in some cases.

### Case Study 1: No Trend or Seasonality

Thedaily female birthsdataset summarizes the daily total female births
in California, USA in
1959. For more information on this dataset, see Chapter 11 where it was
introduced. You can
download the dataset directly from here:

- daily-total-female-births.csv^1

Save the file with the filename `daily-total-female-births.csv` in your
current working
directory. The dataset has one year, or 365 observations. We will use
the first 200 for training
and the remaining 165 as the test set. The complete example grid
searching the daily female
univariate time series forecasting problem is listed below.

```

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

def sarima_forecast(history, config):
order, sorder, trend = config
model = SARIMAX(history, order=order, seasonal_order=sorder,
trend=trend,
enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(history), len(history))
return yhat[0]

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = sarima_forecast(history, cfg)

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

def sarima_configs(seasonal=[0]):
models = list()

p_params = [0, 1, 2]
d_params = [0, 1]
q_params = [0, 1, 2]
t_params = ['n','c','t','ct']
P_params = [0, 1, 2]
D_params = [0, 1]
Q_params = [0, 1, 2]
m_params = seasonal

for p in p_params:


for d in d_params:
for q in q_params:
for t in t_params:
for P in P_params:
for D in D_params:
for Q in Q_params:
for m in m_params:
cfg = [(p,d,q), (P,D,Q,m), t]
models.append(cfg)
return models

if **name** =='**main**':
series = read_csv('daily-total-female-births.csv', header=0,
index_col=0)
data = series.values
n_test = 165
cfg_list = sarima_configs()
scores = grid_search(data, cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example may take a while on modern hardware. Model
configurations and the

RMSE are printed as the models are evaluated. The top three model
configurations and their

error are reported at the end of the run.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.


```
...

> Model[[(2, 1, 2), (1, 0, 1, 0),'ct']] 6.905
>  Model[[(2, 1, 2), (2, 0, 0, 0),'ct']] 7.031
>  Model[[(2, 1, 2), (2, 0, 1, 0),'ct']] 6.985
>  Model[[(2, 1, 2), (1, 0, 2, 0),'ct']] 6.941
>  Model[[(2, 1, 2), (2, 0, 2, 0),'ct']] 7.056
>  done

[(1, 0, 2), (1, 0, 1, 0),'t'] 6.770349800255089
[(0, 1, 2), (1, 0, 2, 0),'ct'] 6.773217122759515
[(2, 1, 1), (2, 0, 2, 0),'ct'] 6.886633191752254

```

We can see that the best result was an RMSE of about 6.77 births. A
naive model achieved
an RMSE of 6.93 births suggesting that the best performing SARIMA model
is skillful on this
problem. We can unpack the configuration of the best performing model as
follows:

- Order: (1, 0, 2)

- Seasonal Order: (1, 0, 1, 0)

- Trend Parameter:‘t’for linear trend

It is surprising that a configuration with some seasonal elements resulted in the lowest error.
I would not have guessed at this configuration and would have likely stuck with an ARIMA
model.

### Case Study 2: Trend

Themonthly shampoo salesdataset summarizes the monthly sales of shampoo
over a three-year
period. For more information on this dataset, see Chapter 11 where it was introduced. You can
download the dataset directly from here:

- monthly-shampoo-sales.csv^2

Save the file with the filename `monthly-shampoo-sales.csv` in your current working di-
rectory. The dataset has three years, or 36 observations. We will use the first 24 for training
and the remaining 12 as the test set. The complete example grid searching the shampoo sales
univariate time series forecasting problem is listed below.

```

# grid search sarima hyperparameters for monthly shampoo sales dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
order, sorder, trend = config
# define model
model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
enforce_stationarity=False, enforce_invertibility=False)
# fit model
model_fit = model.fit(disp=False)
# make one step forecast
yhat = model_fit.predict(len(history), len(history))
return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = sarima_forecast(history, cfg)

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

def sarima_configs(seasonal=[0]):
models = list()

p_params = [0, 1, 2]
d_params = [0, 1]
q_params = [0, 1, 2]
t_params = ['n','c','t','ct']
P_params = [0, 1, 2]
D_params = [0, 1]
Q_params = [0, 1, 2]
m_params = seasonal

for p in p_params:
for d in d_params:
for q in q_params:
for t in t_params:
for P in P_params:
for D in D_params:
for Q in Q_params:
for m in m_params:
cfg = [(p,d,q), (P,D,Q,m), t]
models.append(cfg)
return models

if **name** =='**main**':
series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)
data = series.values
n_test = 12
cfg_list = sarima_configs()
scores = grid_search(data, cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example may take a while on modern hardware. Model
configurations and the
RMSE are printed as the models are evaluated. The top three model
configurations and their
error are reported at the end of the run. A truncated example of the
results from running the
hyperparameter grid search are listed below.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.

```

...

> Model[[(2, 1, 2), (1, 0, 1, 0),'ct']] 68.891
>  Model[[(2, 1, 2), (2, 0, 0, 0),'ct']] 75.406


> Model[[(2, 1, 2), (1, 0, 2, 0),'ct']] 80.908
> Model[[(2, 1, 2), (2, 0, 1, 0),'ct']] 78.734
> Model[[(2, 1, 2), (2, 0, 2, 0),'ct']] 82.958
done

[(0, 1, 2), (2, 0, 2, 0),'t'] 54.767582003072874
[(0, 1, 1), (2, 0, 2, 0),'ct'] 58.69987083057107
[(1, 1, 2), (0, 0, 1, 0),'t'] 58.709089340600094

```

We can see that the best result was an RMSE of about 54.76 sales. A naive model achieved
an RMSE of 95.69 sales on this dataset, meaning that the best performing SARIMA model
is skillful on this problem. We can unpack the configuration of the best performing model as
follows:

```

- Trend Order: (0, 1, 2)

- Seasonal Order: (2, 0, 2, 0)

- Trend Parameter:‘t’(linear trend)

### Case Study 3: Seasonality

Themonthly mean temperaturesdataset summarizes the monthly average air
temperatures in

Nottingham Castle, England from 1920 to 1939 in degrees Fahrenheit. For more information on
this dataset, see Chapter 11 where it was introduced. You can download the dataset directly
from here:

- monthly-mean-temp.csv^3

Save the file with the filename `monthly-mean-temp.csv` in your current
working directory.

The dataset has 20 years, or 240 observations. We will trim the dataset
to the last five years of

data (60 observations) in order to speed up the model evaluation process and use the last year
or 12 observations for the test set.

# trim dataset to 5 years
data = data[-(5*12):]

```
The period of the seasonal component is about one year, or 12 observations. We will use
this as the seasonal period in the call to thesarimaconfigs() function when preparing the
model configurations.


```

# model configs
cfg_list = sarima_configs(seasonal=[0, 12])

```

The complete example grid searching the monthly mean temperature time
series forecasting problem is listed below.

```

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

def sarima_forecast(history, config):
order, sorder, trend = config
model = SARIMAX(history, order=order, seasonal_order=sorder,
trend=trend,
enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(history), len(history))
return yhat[0]

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = sarima_forecast(history, cfg)

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

def sarima_configs(seasonal=[0]):
models = list()

p_params = [0, 1, 2]
d_params = [0, 1]
q_params = [0, 1, 2]
t_params = ['n','c','t','ct']
P_params = [0, 1, 2]
D_params = [0, 1]
Q_params = [0, 1, 2]
m_params = seasonal

for p in p_params:
for d in d_params:
for q in q_params:
for t in t_params:
for P in P_params:
for D in D_params:
for Q in Q_params:
for m in m_params:


cfg = [(p,d,q), (P,D,Q,m), t]
models.append(cfg)
return models

if **name** =='**main**':
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
data = series.values

data = data[-(5*12):]
n_test = 12
cfg_list = sarima_configs(seasonal=[0, 12])
scores = grid_search(data, cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example may take a while on modern hardware. Model
configurations and the
RMSE are printed as the models are evaluated. The top three model
configurations and their
error are reported at the end of the run. A truncated example of the
results from running the
hyperparameter grid search are listed below.
**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider
running the example a few times.


```
...

> Model[[(2, 1, 2), (2, 1, 0, 12),'t']] 4.599
>  Model[[(2, 1, 2), (1, 1, 0, 12),'ct']] 2.477
>  Model[[(2, 1, 2), (2, 0, 0, 12),'ct']] 2.548
>  Model[[(2, 1, 2), (2, 0, 1, 12),'ct']] 2.893
>  Model[[(2, 1, 2), (2, 1, 0, 12),'ct']] 5.404
>  done

[(0, 0, 0), (1, 0, 1, 12),'n'] 1.5577613610905712
[(0, 0, 0), (1, 1, 0, 12),'n'] 1.6469530713847962
[(0, 0, 0), (2, 0, 0, 12),'n'] 1.7314448163607488

```

We can see that the best result was an RMSE of about 1.55 degrees. A
naive model achieved
an RMSE of 1.50 degrees, suggesting that the best performing SARIMA
model is not skillful on
this problem. We can unpack the configuration of the best performing
model as follows:

- Trend Order: (0, 0, 0)

- Seasonal Order: (1, 0, 1, 12)

- Trend Parameter:‘n’(no trend)

As we would expect, the model has no trend component and a 12-month seasonal ARIMA
component.

### Case Study 4: Trend and Seasonality

Themonthly car salesdataset summarizes the monthly car sales in Quebec,
Canada between
where it was introduced.

You can download the dataset directly from here:

- monthly-car-sales.csv^4

Save the file with the filename `monthly-car-sales.csv` in your current
working directory.

The dataset has 9 years, or 108 observations. We will use the last year
or 12 observations as the
test set. The period of the seasonal component could be six months or 12 months. We will try
both as the seasonal period in the call to thesarimaconfigs() function when preparing the
model configurations.

```

# model configs
cfg_list = sarima_configs(seasonal=[0,6,12])

```

The complete example grid searching the monthly car sales time series forecasting problem
is listed below.

```

# grid search sarima hyperparameters for monthly car sales dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
order, sorder, trend = config
# define model
model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
enforce_stationarity=False, enforce_invertibility=False)
# fit model
model_fit = model.fit(disp=False)
# make one step forecast
yhat = model_fit.predict(len(history), len(history))
return yhat[0]

# root mean squared error or rmse

def measure_rmse(actual, predicted):
return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
predictions = list()
train, test = train_test_split(data, n_test)

history = [x for x in train]

for i in range(len(test)):

yhat = sarima_forecast(history, cfg)

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
# remove empty results
scores = [r for r in scores if r[1] != None]
# sort configs by error, asc
scores.sort(key=lambda tup: tup[1])
return scores

def sarima_configs(seasonal=[0]):
models = list()

p_params = [0, 1, 2]
d_params = [0, 1]
q_params = [0, 1, 2]
t_params = ['n','c','t','ct']
P_params = [0, 1, 2]
D_params = [0, 1]
Q_params = [0, 1, 2]
m_params = seasonal

for p in p_params:
for d in d_params:
for q in q_params:
for t in t_params:
for P in P_params:
for D in D_params:
for Q in Q_params:
for m in m_params:
cfg = [(p,d,q), (P,D,Q,m), t]
models.append(cfg)
return models

if **name** =='**main**':
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
data = series.values
n_test = 12
cfg_list = sarima_configs(seasonal=[0,6,12])
scores = grid_search(data, cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example may take a while on modern hardware. Model
configurations and the

RMSE are printed as the models are evaluated. The top three model
configurations and their

error are reported at the end of the run. A truncated example of the
results from running the

hyperparameter grid search are listed below.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider


running the example a few times.

```

> Model[[(2, 1, 2), (2, 0, 2, 12),'ct']] 10710.462
> Model[[(2, 1, 2), (2, 1, 2, 6),'ct']] 2183.568
> Model[[(2, 1, 2), (2, 1, 0, 12),'ct']] 2105.800
> Model[[(2, 1, 2), (2, 1, 1, 12),'ct']] 2330.361
> Model[[(2, 1, 2), (2, 1, 2, 12),'ct']] 31580326686.803
done

[(0, 0, 0), (1, 1, 0, 12),'t'] 1551.8423920342414
[(0, 0, 0), (2, 1, 1, 12),'c'] 1557.334614575545
[(0, 0, 0), (1, 1, 0, 12),'c'] 1559.3276311282675

```

We can see that the best result was an RMSE of about 1,551.84 sales. A naive model achieved
an RMSE of 1,841.15 sales on this problem, suggesting that the best performing SARIMA model
is skillful. We can unpack the configuration of the best performing model as follows:

- Trend Order: (0, 0, 0)

- Seasonal Order: (1, 1, 0, 12)

- Trend Parameter:‘t’(linear trend)

### Extensions

This section lists some ideas for extending the tutorial that you may
wish to explore.

- Data Transforms. Update the framework to support configurable data transforms such
as normalization and standardization.

- Plot Forecast. Update the framework to re-fit a model with the best configuration and
forecast the entire test dataset, then plot the forecast compared to the actual observations
in the test set.

- Tune Amount of History. Update the framework to tune the amount of historical data
used to fit the model (e.g. in the case of the 10 years of max temperature data).


### Further Reading

This section provides more resources on the topic if you are looking to
go deeper.

#### APIs

- Statsmodels Time Series Analysis by State Space Methods.
http://www.statsmodels.org/dev/statespace.html

- statsmodels.tsa.statespace.sarimax.SARIMAXAPI.
http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.
SARIMAX.html

- statsmodels.tsa.statespace.sarimax.SARIMAXResultsAPI.
http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.
SARIMAXResults.html

- Statsmodels SARIMAX Notebook.
http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_
stata.html

#### Articles

- Autoregressive integrated moving average, Wikipedia.
https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

### Summary

In this tutorial, you discovered how to develop a framework for grid
searching all of the SARIMA

model hyperparameters for univariate time series forecasting.
Specifically, you learned:

- How to develop a framework for grid searching SARIMA models from scratch using
walk-forward validation.

- How to grid search SARIMA model hyperparameters for daily time series
data for births.

- How to grid search SARIMA model hyperparameters for monthly time series data for
shampoo sales, car sales and temperature.

#### Next

In the next lab, you will discover how to develop deep learning
models for univariate time

series forecasting problems.
