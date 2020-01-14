<img align="right" src="../logo-small.png">


### How to Develop ETS Models for Univariate Forecasting

Exponential smoothing is a time series forecasting method for univariate data that can be
extended to support data with a systematic trend or seasonal component. It is common practice
to use an optimization process to find the model hyperparameters that result in the exponential
smoothing model with the best performance for a given time series dataset. This practice
applies only to the coefficients used by the model to describe the exponential structure of the
level, trend, and seasonality. It is also possible to automatically optimize other hyperparameters
of an exponential smoothing model, such as whether or not to model the trend and seasonal
component and if so, whether to model them using an additive or multiplicative method.
In this tutorial, you will discover how to develop a framework for grid searching all of the
exponential smoothing model hyperparameters for univariate time series forecasting. After
completing this tutorial, you will know:

- How to develop a framework for grid searching ETS models from scratch using walk-forward
validation.

- How to grid search ETS model hyperparameters for daily time series data for female
births.

- How to grid search ETS model hyperparameters for monthly time series data for shampoo
sales, car sales, and temperature.

Let’s get started.

### Tutorial Overview

This tutorial is divided into five parts; they are:

1.  Develop a Grid Search Framework
2.  Case Study 1: No Trend or Seasonality
3.  Case Study 2: Trend
4.  Case Study 3: Seasonality
5.  Case Study 4: Trend and Seasonality

### Develop a Grid Search Framework

In this section, we will develop a framework for grid searching exponential smoothing model
hyperparameters for a given univariate time series forecasting problem. For more information
on exponential smoothing for time series forecasting, also called ETS, see Chapter 5. We will
use the implementation of Holt-Winters Exponential Smoothing provided by the Statsmodels
library. This model has hyperparameters that control the nature of the exponential performed
for the series, trend, and seasonality, specifically:

- smoothinglevel(alpha): the smoothing coefficient for the level.

- smoothingslope(beta): the smoothing coefficient for the trend.

- smoothingseasonal(gamma): the smoothing coefficient for the seasonal
component.

- dampingslope(phi): the coefficient for the damped trend.

All four of these hyperparameters can be specified when defining the model. If they are not
specified, the library will automatically tune the model and find the optimal values for these
hyperparameters (e.g.optimized=True). There are other hyperparameters that the model will
not automatically tune that you may want to specify; they are:

- trend: The type of trend component, as eitheraddfor additive ormulfor multiplicative.
Modeling the trend can be disabled by setting it toNone.

- damped: Whether or not the trend component should be damped, either
True or False.

- seasonal: The type of seasonal component, as eitheraddfor additive ormulfor multi-
plicative. Modeling the seasonal component can be disabled by setting it toNone.

- seasonalperiods: The number of time steps in a seasonal period, e.g. 12 for 12 months
in a yearly seasonal structure.

- useboxcox: Whether or not to perform a power transform of the series (True/False) or specify the lambda for the transform.

If you know enough about your problem to specify one or more of these parameters, then
you should specify them. If not, you can try grid searching these
parameters. We can start-off
by defining a function that will fit a model with a given configuration and make a one-step
forecast. Theexpsmoothingforecast()below implements this behavior. The function takes
an array or list of contiguous prior observations and a list of configuration parameters used to
configure the model. The configuration parameters in order are: the trend type, the dampening
type, the seasonality type, the seasonal period, whether or not to use a Box-Cox transform, and
whether or not to remove the bias when fitting the model.

```

# one-step Holt Winter's Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
t,d,s,p,b,r = config
# define model
history = array(history)
model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)


# fit model
model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
# make one step forecast
yhat = model_fit.predict(len(history), len(history))
return yhat[0]

```

In this tutorial, we will use the grid searching framework developed in Chapter 11 for tuning
and evaluating naive forecasting methods. One important modification to the framework is the
function used to perform the walk-forward validation of the model namedwalkforwardvalidation().

This function must be updated to call the function for making an ETS
forecast. The updated version of the function is listed below.

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
yhat = exp_smoothing_forecast(history, cfg)
# store forecast in list of predictions
predictions.append(yhat)
# add actual observation to history for the next loop
history.append(test[i])
# estimate prediction error
error = measure_rmse(test, predictions)
return error

```

We’re nearly done. The only thing left to do is to define a list of model configurations to try
for a dataset. We can define this generically. The only parameter we may want to specify is the
periodicity of the seasonal component in the series, if one exists. By default, we will assume no
seasonal component. The expsmoothingconfigs() function below will create a list of model
configurations to evaluate. An optional list of seasonal periods can be specified, and you could
even change the function to specify other elements that you may know about your time series.
In theory, there are 72 possible model configurations to evaluate, but in practice, many will not
be valid and will result in an error that we will trap and ignore.

```
# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
models = list()
# define config lists
t_params = ['add','mul', None]
d_params = [True, False]
s_params = ['add','mul', None]
p_params = seasonal
b_params = [True, False]
r_params = [True, False]
# create config instances


for t in t_params:
for d in d_params:
for s in s_params:
for p in p_params:
for b in b_params:
for r in r_params:
cfg = [t,d,s,p,b,r]
models.append(cfg)
return models

```

We now have a framework for grid searching triple exponential smoothing model hyperparameters via one-step walk-forward validation. It is generic and will work for any in-memory
univariate time series provided as a list or NumPy array. We can make sure all the pieces work
together by testing it on a contrived 10-step dataset. The complete example is listed below.

```

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array

def exp_smoothing_forecast(history, config):
t,d,s,p,b,r = config

history = array(history)
model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s,
seasonal_periods=p)

model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)

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
# fit model and make forecast for history
yhat = exp_smoothing_forecast(history, cfg)
# store forecast in list of predictions
predictions.append(yhat)
# add actual observation to history for the next loop
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

def exp_smoothing_configs(seasonal=[None]):
models = list()

t_params = ['add','mul', None]
d_params = [True, False]
s_params = ['add','mul', None]


p_params = seasonal
b_params = [True, False]
r_params = [True, False]
# create config instances
for t in t_params:
for d in d_params:
for s in s_params:
for p in p_params:
for b in b_params:
for r in r_params:
cfg = [t,d,s,p,b,r]
models.append(cfg)
return models

if __name__ =='__main__':
# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# data split
n_test = 4
# model configs
cfg_list = exp_smoothing_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example first prints the contrived time series dataset. Next, the model
configurations and their errors are reported as they are evaluated. Finally, the configurations
and the error for the top three configurations are reported.

```
[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

> Model[[None, False, None, None, True, True]] 1.380
> Model[[None, False, None, None, True, False]] 10.000
> Model[[None, False, None, None, False, True]] 2.563
> Model[[None, False, None, None, False, False]] 10.000
done

[None, False, None, None, True, True] 1.379824445857423
[None, False, None, None, False, True] 2.5628662672606612
[None, False, None, None, False, False] 10.0

```

We do not report the model parameters optimized by the model itself. It is assumed that
you can achieve the same result again by specifying the broader
hyperparameters and allow
the library to find the same internal parameters. You can access these internal parameters
by refitting a standalone model with the same configuration and printing the contents of the
paramsattribute on the model fit; for example:

```
# access model parameters
print(model_fit.params)

```

Now that we have a robust framework for grid searching ETS model hyperparameters, let’s
test it out on a suite of standard univariate time series datasets. The datasets were chosen for
demonstration purposes; I am not suggesting that an ETS model is the best approach for each
dataset, and perhaps an SARIMA or something else would be more appropriate in some cases.

### Case Study 1: No Trend or Seasonality

Thedaily female birthsdataset summarizes the daily total female births
in California, USA in
1959. For more information on this dataset, see Chapter 11 where it was
introduced. You can

download the dataset directly from here:

- daily-total-female-births.csv^1

Save the file with the filename `daily-total-female-births.csv` in your current working
directory. The dataset has one year, or 365 observations. We will use the first 200 for training
and the remaining 165 as the test set. The complete example grid searching the daily female
univariate time series forecasting problem is listed below.

```

# grid search ets models for daily female births
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from numpy import array

# one-step Holt Winters Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
t,d,s,p,b,r = config
# define model
history = array(history)
model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
# fit model
model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
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

yhat = exp_smoothing_forecast(history, cfg)

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
# sort configs by error, asc
scores.sort(key=lambda tup: tup[1])
return scores

def exp_smoothing_configs(seasonal=[None]):
models = list()

t_params = ['add','mul', None]
d_params = [True, False]
s_params = ['add','mul', None]
p_params = seasonal
b_params = [True, False]
r_params = [True, False]

for t in t_params:
for d in d_params:
for s in s_params:
for p in p_params:
for b in b_params:
for r in r_params:
cfg = [t,d,s,p,b,r]
models.append(cfg)
return models

if **name** =='**main**':

series = read_csv('daily-total-female-births.csv', header=0,
index_col=0)
data = series.values

n_test = 165

cfg_list = exp_smoothing_configs()

scores = grid_search(data[:,0], cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example may take a few minutes as fitting each ETS model can take about a
minute on modern hardware. Model configurations and the RMSE are printed
as the models
are evaluated. The top three model configurations and their error are
reported at the end of the
run. A truncated example of the results from running the hyperparameter
grid search are listed below.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.

```
...

> Model[['mul', False, None, None, True, False]] 6.985
> Model[[None, False, None, None, True, True]] 7.169


> Model[[None, False, None, None, True, False]] 7.212
> Model[[None, False, None, None, False, True]] 7.117
> Model[[None, False, None, None, False, False]] 7.126
done

['mul', False, None, None, True, True] 6.960703917145126
['mul', False, None, None, True, False] 6.984513598720297
['add', False, None, None, True, True] 7.081359856193836

```

We can see that the best result was an RMSE of about 6.96 births. A naive model achieved
an RMSE of 6.93 births, meaning that the best performing ETS model is not skillful on this
problem. We can unpack the configuration of the best performing model as follows:

- Trend: Multiplicative

- Damped: False

- Seasonal: None

- Seasonal Periods: None

- Box-Cox Transform: True

- Remove Bias: True

What is surprising is that a model that assumed an multiplicative trend performed better
than one that didn’t. We would not know that this is the case unless we threw out assumptions
and grid searched models.

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
# grid search ets models for monthly shampoo sales
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from numpy import array

def exp_smoothing_forecast(history, config):
t,d,s,p,b,r = config

history = array(history)
model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s,
seasonal_periods=p)

model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)

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

yhat = exp_smoothing_forecast(history, cfg)

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

def exp_smoothing_configs(seasonal=[None]):
models = list()

t_params = ['add','mul', None]
d_params = [True, False]
s_params = ['add','mul', None]
p_params = seasonal
b_params = [True, False]
r_params = [True, False]

for t in t_params:
for d in d_params:
for s in s_params:
for p in p_params:
for b in b_params:
for r in r_params:
cfg = [t,d,s,p,b,r]
models.append(cfg)
return models

if **name** =='**main**':

series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)
data = series.values

n_test = 12


cfg_list = exp_smoothing_configs()
# grid search
scores = grid_search(data[:,0], cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example is fast given there are a small number of
observations. Model configura-
tions and the RMSE are printed as the models are evaluated. The top
three model configurations
and their error are reported at the end of the run. A truncated example
of the results from running the hyperparameter grid search are listed below.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider
running the example a few times.

```

...

> Model[['mul', True, None, None, False, False]] 102.152
>  Model[['mul', False, None, None, False, True]] 86.406
>  Model[['mul', False, None, None, False, False]] 83.747
>  Model[[None, False, None, None, False, True]] 99.416
>  Model[[None, False, None, None, False, False]] 108.031
>  done

['mul', False, None, None, False, False] 83.74666940175238
['mul', False, None, None, False, True] 86.40648953786152
['mul', True, None, None, False, True] 95.33737598817238

```


We can see that the best result was an RMSE of about 83.74 sales. A
naive model achieved
an RMSE of 95.69 sales on this dataset, meaning that the best performing
ETS model is skillful
on this problem. We can unpack the configuration of the best performing
model as follows:

- Trend: Multiplicative

- Damped: False

- Seasonal: None

- Seasonal Periods: None

- Box-Cox Transform: False

- Remove Bias: False


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

```

# trim dataset to 5 years
data = data[-(5*12):]

```

The period of the seasonal component is about one year, or 12 observations. We will use this
as the seasonal period in the call to the expsmoothingconfigs() function when preparing
the model configurations.


```

# model configs
cfg_list = exp_smoothing_configs(seasonal=[0, 12])

```

The complete example grid searching the monthly mean temperature time series forecasting
problem is listed below.

```

# grid search ets hyperparameters for monthly mean temp dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from numpy import array

# one-step Holt Winters Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
t,d,s,p,b,r = config
# define model
history = array(history)
model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
# fit model
model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
# make one step forecast
yhat = model_fit.predict(len(history), len(history))

(^3)
https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-mean-temp.csv


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

yhat = exp_smoothing_forecast(history, cfg)

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
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
scores = executor(tasks)
else:
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
# remove empty results
scores = [r for r in scores if r[1] != None]
# sort configs by error, asc
scores.sort(key=lambda tup: tup[1])
return scores

def exp_smoothing_configs(seasonal=[None]):
models = list()

t_params = ['add','mul', None]
d_params = [True, False]
s_params = ['add','mul', None]
p_params = seasonal
b_params = [True, False]
r_params = [True, False]

for t in t_params:
for d in d_params:
for s in s_params:
for p in p_params:
for b in b_params:
for r in r_params:
cfg = [t,d,s,p,b,r]
models.append(cfg)
return models

if **name** =='**main**':

series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
data = series.values

data = data[-(5*12):]

n_test = 12

cfg_list = exp_smoothing_configs(seasonal=[0,12])

scores = grid_search(data[:,0], cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example is relatively slow given the large amount of data.
Model configurations
and the RMSE are printed as the models are evaluated. The top three
model configurations
and their error are reported at the end of the run. A truncated example
of the results from
running the hyperparameter grid search are listed below.


**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```

> Model[['mul', True, None, 12, False, False]] 4.593
> Model[['mul', False,'add', 12, True, True]] 4.230
> Model[['mul', False,'add', 12, True, False]] 4.157
> Model[['mul', False,'add', 12, False, True]] 1.538
> Model[['mul', False,'add', 12, False, False]] 1.520
done

[None, False,'add', 12, False, False] 1.5015527325330889
[None, False,'add', 12, False, True] 1.5015531225114707
[None, False,'mul', 12, False, False] 1.501561363221282

```

We can see that the best result was an RMSE of about 1.50 degrees. This is the same RMSE
found by a naive model on this problem, suggesting that the best ETS model sits on the border
of being unskillful. We can unpack the configuration of the best performing model as follows:

- Trend: None

- Damped: False

- Seasonal: Additive

- Seasonal Periods: 12

- Box-Cox Transform: False

- Remove Bias: False

### Case Study 4: Trend and Seasonality

Themonthly car salesdataset summarizes the monthly car sales in Quebec,
Canada between

where it was introduced.

You can download the dataset directly from here:

- monthly-car-sales.csv^4

Save the file with the filename `monthly-car-sales.csv` in your current
working directory.

The dataset has 9 years, or 108 observations. We will use the last year
or 12 observations as
the test set. The period of the seasonal component could be six months or 12 months. We
will try both as the seasonal period in the call to
the expsmoothingconfigs() function when
preparing the model configurations.


```
cfg_list = exp_smoothing_configs(seasonal=[0,6,12])

```

The complete example grid searching the monthly car sales time series
forecasting problem is listed below.

```

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from numpy import array

def exp_smoothing_forecast(history, config):
t,d,s,p,b,r = config

history = array(history)
model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s,
seasonal_periods=p)

model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)

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

yhat = exp_smoothing_forecast(history, cfg)

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

def exp_smoothing_configs(seasonal=[None]):
models = list()

t_params = ['add','mul', None]
d_params = [True, False]
s_params = ['add','mul', None]
p_params = seasonal
b_params = [True, False]
r_params = [True, False]

for t in t_params:
for d in d_params:
for s in s_params:
for p in p_params:


for b in b_params:
for r in r_params:
cfg = [t,d,s,p,b,r]
models.append(cfg)
return models

if **name** =='**main**':

series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
data = series.values

n_test = 12

cfg_list = exp_smoothing_configs(seasonal=[0,6,12])

scores = grid_search(data[:,0], cfg_list, n_test)
print('done')

for cfg, error in scores[:3]:
print(cfg, error)

```

Running the example is slow given the large amount of data. Model
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

> Model[['mul', True,'add', 6, False, False]] 3745.142
>  Model[['mul', True,'add', 12, True, True]] 2203.354
>  Model[['mul', True,'add', 12, True, False]] 2284.172
>  Model[['mul', True,'add', 12, False, True]] 2842.605
>  Model[['mul', True,'add', 12, False, False]] 2086.899

done

['add', False,'add', 12, False, True] 1672.5539372356582
['add', False,'add', 12, False, False] 1680.845043013083
['add', True,'add', 12, False, False] 1696.1734099400082

```

We can see that the best result was an RMSE of about 1,672 sales. A
naive model achieved
an RMSE of 1841.15 sales on this problem, suggesting that the best
performing ETS model is skillful. We can unpack the configuration of the best performing model
as follows:


- Trend: Additive

- Damped: False

- Seasonal: Additive

- Seasonal Periods: 12

- Box-Cox Transform: False

- Remove Bias: True

This is a little surprising as I would have guessed that a six-month seasonal model would be
the preferred approach.

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

- statsmodels.tsa.holtwinters.ExponentialSmoothingAPI.

- statsmodels.tsa.holtwinters.HoltWintersResultsAPI.


#### Articles

- Exponential smoothing, Wikipedia.
https://en.wikipedia.org/wiki/Exponential_smoothing

### Summary

In this tutorial, you discovered how to develop a framework for grid searching all of the
exponential smoothing model hyperparameters for univariate time series forecasting. Specifically,

you learned:

- How to develop a framework for grid searching ETS models from scratch using walk-forward
validation.

- How to grid search ETS model hyperparameters for daily time series
data for births.

- How to grid search ETS model hyperparameters for monthly time series data for shampoo
sales, car sales and temperature.

#### Next

In the next lab, you will discover how to develop autoregressive models for univariate time
series forecasting problems.
