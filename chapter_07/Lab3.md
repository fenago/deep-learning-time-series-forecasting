<img align="right" src="../logo-small.png">


### How to Develop MLPs for Time Series Forecasting (Part 2)

Multilayer Perceptrons, or MLPs for short, can be applied to time series
forecasting. A challenge
with using MLPs for time series forecasting is in the preparation of the
data. Specifically, lag
observations must be flattened into feature vectors. In this tutorial, you will discover how to
develop a suite of MLP models for a range of standard time series forecasting problems. The
objective of this tutorial is to provide standalone examples of each model on each type of time
series problem as a template that you can copy and adapt for your specific time series forecasting
problem. In this tutorial, you will discover how to develop a suite of Multilayer Perceptron
models for a range of standard time series forecasting problems. After completing this tutorial,
you will know:

- How to develop MLP models for univariate time series forecasting.

- How to develop MLP models for multivariate time series forecasting.

- How to develop MLP models for multi-step time series forecasting.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-time-series-forecasting` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab3_MLPs_Time_Series_Forecasting_Part2`

# Vector-Output MLP Model

We are now ready to fit an MLP model on this data. As with the previous
case of multivariate
input, we must flatten the three dimensional structure of the input data samples to a two
dimensional structure of [samples, features], where lag observations are treated as features
by the model.

```
# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

```

The model output will be a vector, with one element for each of the three different time
series.

```
# determine the number of outputs
n_output = y.shape[1]

```

We can now define our model, using the flattened vector length for the input layer and the
number of time series as the vector length when making a prediction.

```
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')

```

We can predict the next value in each of the three parallel series by providing an input of
three time steps for each series.

```
70, 75, 145
80, 85, 165
90, 95, 185

```

The shape of the input for making a single prediction must be 1 sample, 3 time steps and 3
features, or[1, 3, 3]. Again, we can flatten this to[1, 6]to meet the expectations of the
model. We would expect the vector output to be:

```
[100, 105, 205]

```


```
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)

```

We can tie all of this together and demonstrate an MLP for multivariate output time series
forecasting below.

```
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
end_ix = i + n_steps
if end_ix > len(sequences)-1:
break
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in
range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps = 3
X, y = split_sequences(dataset, n_steps)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
n_output = y.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `08_mlp_multivariate_time_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model, and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[100.95039 107.541306 206.81033 ]]

```

### Multi-output MLP Model

As with multiple input series, there is another, more elaborate way to
model the problem. Each
output series can be handled by a separate output MLP model. We can refer to this as a
multi-output MLP model. It may offer more flexibility or better performance depending on the
specifics of the problem that is being modeled. This type of model can be defined in Keras
using the Keras functional API. First, we can define the input model as an MLP with an input
layer that expects flattened feature vectors.

```
# define model
visible = Input(shape=(n_input,))
dense = Dense(100, activation='relu')(visible)

```

We can then define one output layer for each of the three series that we wish to forecast,
where each output submodel will forecast a single time step.

```
# define output 1
output1 = Dense(1)(dense)
# define output 2
output2 = Dense(1)(dense)
# define output 2
output3 = Dense(1)(dense)

```
We can then tie the input and output layers together into a single model.

```
# tie together
model = Model(inputs=visible, outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')

```
To make the model architecture clear, the schematic below clearly shows the three separate
output layers of the model and the input and output shapes of each layer.

![](./images/88-3.png)

When training the model, it will require three separate output arrays per sample. We can
achieve this by converting the output training data that has the shape[7, 3]to three arrays
with the shape[7, 1].

```
# separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))

```

These arrays can be provided to the model during training.

```
# fit model
model.fit(X, [y1,y2,y3], epochs=2000, verbose=0)

```

Tying all of this together, the complete example is listed below.


```
# multivariate output mlp example
from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences)-1:
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# flatten input
n_input = X.shape[1] * X.shape[2]


X = X.reshape((X.shape[0], n_input))
# separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))
# define model
visible = Input(shape=(n_input,))
dense = Dense(100, activation='relu')(visible)
# define output 1
output1 = Dense(1)(dense)
# define output 2
output2 = Dense(1)(dense)
# define output 2
output3 = Dense(1)(dense)
# tie together
model = Model(inputs=visible, outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, [y1,y2,y3], epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `09_multi-output_mlp_multivariate_time_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model, and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[array([[100.86121]], dtype=float32),
array([[105.14738]], dtype=float32),
array([[205.97507]], dtype=float32)]

```

### Multi-step MLP Models

In practice, there is little difference to the MLP model in predicting a vector output that
represents different output variables (as in the previous example) or a vector output that
represents multiple time steps of one variable. Nevertheless, there are subtle and important
differences in the way the training data is prepared. In this section, we will demonstrate the
case of developing a multi-step forecast model using a vector model. Before we look at the
specifics of the model, let's first look at the preparation of data for multi-step forecasting.

#### Data Preparation

As with one-step forecasting, a time series used for multi-step time series forecasting must be
split into samples with input and output components. Both the input and output components
will be comprised of multiple time steps and may or may not have the same number of steps.
For example, given the univariate time series:

```
[10, 20, 30, 40, 50, 60, 70, 80, 90]

```
We could use the last three time steps as input and forecast the next two time steps. The
first sample would look as follows:


```
Input:
[10, 20, 30]

Output:
[40, 50]

```

The splitsequence() function below implements this behavior and will split a given
univariate time series into samples with a specified number of input and output time steps.

```
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequence)):
# find the end of this pattern
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out
# check if we are beyond the sequence
if out_end_ix > len(sequence):
break
# gather input and output parts of the pattern
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

```

We can demonstrate this function on the small contrived dataset. The complete example is
listed below.

```
# multi-step data preparation
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequence)):
# find the end of this pattern
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out
# check if we are beyond the sequence
if out_end_ix > len(sequence):


break
# gather input and output parts of the pattern
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `10_data_prep_multi_step_forecasting.ipynb` in jupterLab UI and run jupyter notebook.

Running the example splits the univariate series into input and output time steps and prints
the input and output components of each.

```
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]

```

Now that we know how to prepare data for multi-step forecasting, let's look at an MLP
model that can learn this mapping.

#### Vector Output Model

The MLP can output a vector directly that can be interpreted as a
multi-step forecast. This
approach was seen in the previous section were one time step of each output time series was
forecasted as a vector. With the number of input and output steps specified in thenstepsin
andnstepsoutvariables, we can define a multi-step time-series forecasting model.

```
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

```

The model can make a prediction for a single sample. We can predict the next two steps
beyond the end of the dataset by providing the input:

```
[70, 80, 90]

```

We would expect the predicted output to be:

```
[100, 110]

```

As expected by the model, the shape of the single sample of input data when making the
prediction must be [1, 3] for the 1 sample and 3 time steps (features) of
the input and the
single feature.


```
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)

```

Tying all of this together, the MLP for multi-step forecasting with a
univariate time series is
listed below.

```
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
def split_sequence(sequence, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequence)):
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out
if out_end_ix > len(sequence):
break
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)


print(yhat)

```

##### Run Notebook
Click notebook `11_mlp_multi_step_forecast.ipynb` in jupterLab UI and run jupyter notebook.

Running the example forecasts and prints the next two time steps in the
sequence.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.


```
[[102.572365 113.88405 ]]

```

#### Multivariate Multi-step MLP Models

In the previous sections, we have looked at univariate, multivariate, and multi-step time series
forecasting. It is possible to mix and match the different types of MLP models presented so
far for the different problems. This too applies to time series forecasting problems that involve
multivariate and multi-step forecasting, but it may be a little more challenging, particularly in
preparing the data and defining the shape of inputs and outputs for the model. In this section,
we will look at short examples of data preparation and modeling for multivariate multi-step
time series forecasting as a template to ease this challenge, specifically:
1. Multiple Input Multi-step Output.
2. Multiple Parallel Input and Multi-step Output.

Perhaps the biggest stumbling block is in the preparation of data, so this is where we will
focus our attention.

#### Multiple Input Multi-step Output
There are those multivariate time series forecasting problems where the output series is separate
but dependent upon the input time series, and multiple time steps are required for the output
series. For example, consider our multivariate time series from a prior section:

```
[[ 10 15 25]
[ 20 25 45]
[ 30 35 65]
[ 40 45 85]
[ 50 55 105]
[ 60 65 125]
[ 70 75 145]
[ 80 85 165]
[ 90 95 185]]
```

We may use three prior time steps of each of the two input time series to predict two time
steps of the output time series.


```
Input:

10, 15

20, 25

30, 35

Output:

65
85

```

The splitsequences() function below implements this behavior.


```
def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out-1
if out_end_ix > len(sequences):
break
seq_x, seq_y = sequences[i:end_ix, :-1],
sequences[end_ix-1:out_end_ix, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

```

We can demonstrate this on our contrived dataset. The complete example
is listed below.


```
from numpy import array
from numpy import hstack
def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out-1
if out_end_ix > len(sequences):
break
seq_x, seq_y = sequences[i:end_ix, :-1],
sequences[end_ix-1:out_end_ix, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])


out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `12_prepare_data_multi_step_dependent_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first prints the shape of the prepared training data. We can see that
the shape of the input portion of the samples is three-dimensional, comprised of six samples,
with three time steps and two variables for the two input time series. The output portion of the
samples is two-dimensional for the six samples and the two time steps for each sample to be
predicted. The prepared samples are then printed to confirm that the data was prepared as we
specified.

```
(6, 3, 2) (6, 2)

[[10 15]
[20 25]
[30 35]] [65 85]
[[20 25]
[30 35]
[40 45]] [ 85 105]
[[30 35]
[40 45]
[50 55]] [105 125]
[[40 45]
[50 55]
[60 65]] [125 145]
[[50 55]
[60 65]
[70 75]] [145 165]
[[60 65]
[70 75]
[80 85]] [165 185]

```

We can now develop an MLP model for multi-step predictions using a vector output. The
complete example is listed below.

```
# multivariate multi-step mlp example
from numpy import array
from numpy import hstack
from keras.models import Sequential


from keras.layers import Dense
def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out-1
if out_end_ix > len(sequences):
break
seq_x, seq_y = sequences[i:end_ix, :-1],
sequences[end_ix-1:out_end_ix, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in
range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps_in, n_steps_out = 3, 2
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `13_mlp_multi_step_dependent_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example fits the model and predicts the next two time steps of the output
sequence beyond the dataset. We would expect the next two steps to be [185, 205].

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.

```
[[186.53822 208.41725]]

```

A problem with parallel time series may require the prediction of
multiple time steps of each
time series. For example, consider our multivariate time series from a prior section:

```
[[ 10 15 25]
[ 20 25 45]
[ 30 35 65]
[ 40 45 85]
[ 50 55 105]
[ 60 65 125]
[ 70 75 145]
[ 80 85 165]
[ 90 95 185]]

```
We may use the last three time steps from each of the three time series as input to the model
and predict the next time steps of each of the three time series as output. The first sample in
the training dataset would be the following.

```
Input:

10, 15, 25
20, 25, 45
30, 35, 65

Output:
40, 45, 85
50, 55, 105

```

The splitsequences() function below implements this behavior.

```
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out
# check if we are beyond the dataset
if out_end_ix > len(sequences):
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
X.append(seq_x)
y.append(seq_y)


return array(X), array(y)

```

We can demonstrate this function on the small contrived dataset. The
complete example is listed below.


```
from numpy import array
from numpy import hstack
def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out
if out_end_ix > len(sequences):
break
seq_x, seq_y = sequences[i:end_ix, :],
sequences[end_ix:out_end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in
range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps_in, n_steps_out = 3, 2
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `14_prepare_data_multi_step_multivariate_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first prints the shape of the prepared training dataset. We can see
that both the input (X) and output (Y ) elements of the dataset are three dimensional for the
number of samples, time steps, and variables or parallel time series respectively. The input and
output elements of each series are then printed side by side so that we can confirm that the
data was prepared as we expected.

```
(5, 3, 3) (5, 2, 3)
[[10 15 25]
[20 25 45]
[30 35 65]] [[ 40 45 85]
[ 50 55 105]]
[[20 25 45]
[30 35 65]
[40 45 85]] [[ 50 55 105]
[ 60 65 125]]
[[ 30 35 65]
[ 40 45 85]
[ 50 55 105]] [[ 60 65 125]
[ 70 75 145]]
[[ 40 45 85]
[ 50 55 105]
[ 60 65 125]] [[ 70 75 145]
[ 80 85 165]]
[[ 50 55 105]
[ 60 65 125]
[ 70 75 145]] [[ 80 85 165]
[ 90 95 185]]
```

We can now develop an MLP model to make multivariate multi-step forecasts. In addition
to flattening the shape of the input data, as we have in prior examples, we must also flatten the
three-dimensional structure of the output data. This is because the MLP model is only capable
of taking vector inputs and outputs.


```
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))

```


The complete example is listed below.

```
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):
end_ix = i + n_steps_in
out_end_ix = end_ix + n_steps_out
if out_end_ix > len(sequences):
break

seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in
range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps_in, n_steps_out = 3, 2
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `15_mlp_multi_step_multivariate_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example fits the model and predicts the values for each of
the three time steps
for the next two time steps beyond the end of the dataset. We would
expect the values for these
series and time steps to be as follows:

```
90, 95, 185
100, 105, 205

```

We can see that the model forecast gets reasonably close to the expected
values.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.

```

[[ 91.28376 96.567 188.37575 100.54482 107.9219 208.108 ]]


```

## Exercises

This section lists some ideas for extending the tutorial that you may
wish to explore.

- Problem Differences. Explain the main changes to the MLP required when modeling
each of univariate, multivariate and multi-step time series forecasting problems.

- Experiment. Select one example and modify it to work with your own small contrived
dataset.

- Develop Framework. Use the examples in this chapter as the basis for a framework for
automatically developing an MLP model for a given time series forecasting problem.


### Further Reading

This section provides more resources on the topic if you are looking to
go deeper.

APIs

- Keras: The Python Deep Learning library.
https://keras.io/

- Getting started with the Keras Sequential model.
https://keras.io/getting-started/sequential-model-guide/

- Getting started with the Keras functional API.
https://keras.io/getting-started/functional-api-guide/

- Keras Sequential Model API.
https://keras.io/models/sequential/

- Keras Core Layers API.
https://keras.io/layers/core/


### Summary

In this tutorial, you discovered how to develop a suite of Multilayer
Perceptron, or MLP, models

for a range of standard time series forecasting problems. Specifically,
you learned:

- How to develop MLP models for univariate time series forecasting.

- How to develop MLP models for multivariate time series forecasting.

- How to develop MLP models for multi-step time series forecasting.

#### Next

In the next lab, you will discover how to develop Convolutional
Neural Network models for time series forecasting.
