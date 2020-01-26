<img align="right" src="../logo-small.png">


### How to Develop LSTMs for Time Series Forecasting (Part 2)

Long Short-Term Memory networks, or LSTMs for short, can be applied to time series forecasting.
There are many types of LSTM models that can be used for each specific type of time series
forecasting problem. In this tutorial, you will discover how to develop a suite of LSTM models
for a range of standard time series forecasting problems. The objective of this tutorial is to
provide standalone examples of each model on each type of time series problem as a template
that you can copy and adapt for your specific time series forecasting problem.
After completing this tutorial, you will know:

- How to develop LSTM models for univariate time series forecasting.

- How to develop LSTM models for multivariate time series forecasting.

- How to develop LSTM models for multi-step time series forecasting.

Let' s get started.

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-time-series-forecasting` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab7_LSTMs_Time_Series_Forecasting_Part2`


#### Multiple Parallel Series

An alternate time series problem is the case where there are multiple
parallel time series and a
value must be predicted for each. For example, given the data from the
previous section:

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

We may want to predict the value for each of the three time series for the next time step. This
might be referred to as multivariate forecasting. Again, the data must be split into input/output
samples in order to train a model. The first sample of this dataset would be:


```
Input:

10, 15, 25

20, 25, 45

30, 35, 65

```


```
Output:

40, 45, 85

```

The splitsequences() function below will split multiple parallel time
series with rows for
time steps and one series per column into the required input/output
shape.

```
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

```

We can demonstrate this on the contrived problem; the complete example
is listed below.

```
from numpy import array
from numpy import hstack

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
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `10_multivariate_parallel_series_dataset.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first prints the shape of the prepared `X` and `y` components. The
shape ofXis three-dimensional, including the number of samples (6), the number of time steps
chosen per sample (3), and the number of parallel time series or features (3). The shape ofy
is two-dimensional as we might expect for the number of samples (6) and the number of time
variables per sample to be predicted (3). The data is ready to use in an
LSTM model that
expects three-dimensional input and two-dimensional output shapes for the `X` and `y` components
of each sample. Then, each of the samples is printed showing the input and output components
of each sample.


```
(6, 3, 3) (6, 3)

[[10 15 25]
[20 25 45]
[30 35 65]] [40 45 85]
[[20 25 45]
[30 35 65]
[40 45 85]] [ 50 55 105]
[[ 30 35 65]
[ 40 45 85]
[ 50 55 105]] [ 60 65 125]
[[ 40 45 85]
[ 50 55 105]
[ 60 65 125]] [ 70 75 145]
[[ 50 55 105]
[ 60 65 125]
[ 70 75 145]] [ 80 85 165]
[[ 60 65 125]
[ 70 75 145]
[ 80 85 165]] [ 90 95 185]

```

We are now ready to fit an LSTM model on this data. Any of the varieties of LSTMs in the
previous section can be used, such as a Vanilla, Stacked, Bidirectional, CNN, or ConvLSTM
model. We will use a Stacked LSTM where the number of time steps and parallel series (features)
are specified for the input layer via the input shape argument. The number of parallel series is
also used in the specification of the number of values to predict by the model in the output
layer; again, this is three.


```
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True,
input_shape=(n_steps,
n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

```

We can predict the next value in each of the three parallel series by providing an input of
three time steps for each series.

```
80, 85, 165
90, 95, 185

```
The shape of the input for making a single prediction must be 1 sample,
3 time steps, and 3 features, or[1, 3, 3].

```
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

```

We would expect the vector output to be:

```
[100, 105, 205]

```
We can tie all of this together and demonstrate a Stacked LSTM for multivariate output
time series forecasting below.

```
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
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
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps,
n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=400, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```
##### Run Notebook
Click notebook `11_stacked_lstm_multivariate_parallel_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model, and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[101.76599 108.730484 206.63577 ]]

```

For an example of LSTM models developed for a multivariate time series forecasting problem,
see Chapter 20.

### Multi-step LSTM Models

A time series forecasting problem that requires a prediction of multiple
time steps into the

future can be referred to as multi-step time series forecasting. Specifically, these are problems

where the forecast horizon or interval is more than one time step. There
are two main types of

LSTM models that can be used for multi-step forecasting; they are:


1.  Vector Output Model
2.  Encoder-Decoder Model

<!-- -->

Before we look at these models, let's first look at the preparation of data for multi-step
forecasting.

#### Data Preparation

As with one-step forecasting, a time series used for multi-step time
series forecasting must be

split into samples with input and output components. Both the input and
output components

will be comprised of multiple time steps and may or may not have the
same number of steps.

For example, given the univariate time series:

```
[10, 20, 30, 40, 50, 60, 70, 80, 90]

```
We could use the last three time steps as input and forecast the next two time steps. The
first sample would look as follows:

```
Input:
[10, 20, 30]

```


```
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
from numpy import array

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

for i in range(len(X)):
print(X[i], y[i])

```


##### Run Notebook
Click notebook `12_multi_step_series_dataset.ipynb` in jupterLab UI and run jupyter notebook.

Running the example splits the univariate series into input and output
time steps and prints the input and output components of each.

```
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]

```

Now that we know how to prepare data for multi-step forecasting, let's
look at some LSTM
models that can learn this mapping.

#### Vector Output Model

Like other types of neural network models, the LSTM can output a vector
directly that can
be interpreted as a multi-step forecast. This approach was seen in the
previous section were
one time step of each output time series was forecasted as a vector. As
with the LSTMs for
univariate data in a prior section, the prepared samples must first be
reshaped. The LSTM
expects data to have a three-dimensional structure of [samples,
timesteps, features], and
in this case, we only have one feature so the reshape is
straightforward.


```
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

```

With the number of input and output steps specified in
the n_steps_in and n_steps_out
variables, we can define a multi-step time-series forecasting model. Any
of the presented LSTM
model types could be used, such as Vanilla, Stacked, Bidirectional, CNN-LSTM, or ConvLSTM.
Below defines a Stacked LSTM for multi-step forecasting.

```
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in,
n_features)))
model.add(LSTM(100, activation='relu'))
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
prediction must be [1, 3, 1]for the 1 sample, 3 time steps of the input, and the single feature.


```
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)

```

Tying all of this together, the Stacked LSTM for multi-step forecasting with a univariate
time series is listed below.

```
# univariate multi-step vector-output stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

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
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in,
n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=50, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `13_stacked_lstm_multi_step.ipynb` in jupterLab UI and run jupyter notebook.

Running the example forecasts and prints the next two time steps in the
sequence.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[100.98096 113.28924]]

```

#### Encoder-Decoder Model

A model specifically developed for forecasting variable length output
sequences is called the
Encoder-Decoder LSTM. The model was designed for prediction problems where there are
both input and output sequences, so-called sequence-to-sequence, or seq2seq problems, such
as translating text from one language to another. This model can be used for multi-step time
series forecasting. As its name suggests, the model is comprised of two sub-models: the encoder
and the decoder.

The encoder is a model responsible for reading and interpreting the input sequence. The
output of the encoder is a fixed length vector that represents the model' s interpretation of the
sequence. The encoder is traditionally a Vanilla LSTM model, although other encoder models
can be used such as Stacked, Bidirectional, and CNN models.

```
# define encoder model
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))

```
The decoder uses the output of the encoder as an input. First, the fixed-length output of
the encoder is repeated, once for each required time step in the output sequence.

```
# repeat encoding
model.add(RepeatVector(n_steps_out))

```

This sequence is then provided to an LSTM decoder model. The model must output a value
for each value in the output time step, which can be interpreted by a single output model.

```
# define decoder model
model.add(LSTM(100, activation='relu', return_sequences=True))

```
We can use the same output layer or layers to make each one-step prediction in the output
sequence. This can be achieved by wrapping the output part of the model in aTimeDistributed
wrapper.

```
# define model output
model.add(TimeDistributed(Dense(1)))

```
The full definition for an Encoder-Decoder model for multi-step time series forecasting is
listed below.

```
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

```

As with other LSTM models, the input data must be reshaped into the expected three-
dimensional shape of [samples, timesteps, features].

```
# reshape input training data
X = X.reshape((X.shape[0], X.shape[1], n_features))

```

In the case of the Encoder-Decoder model, the output, orypart, of the training dataset
must also have this shape. This is because the model will predict a given number of time steps
with a given number of features for each input sample.


```
y = y.reshape((y.shape[0], y.shape[1], n_features))

```

The complete example of an Encoder-Decoder LSTM for multi-step time series forecasting
is listed below.

```
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

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

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in,
n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=100, verbose=0)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


```

##### Run Notebook
Click notebook `14_encoder_decoder_lstm_multi_step.ipynb` in jupterLab UI and run jupyter notebook.

Running the example forecasts and prints the next two time steps in the
sequence.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.


```
[[[101.9736

[116.213615]]]

```

### Multivariate Multi-step LSTM Models

In the previous sections, we have looked at univariate, multivariate, and multi-step time series
forecasting. It is possible to mix and match the different types of LSTM models presented so
far for the different problems. This too applies to time series forecasting problems that involve
multivariate and multi-step forecasting, but it may be a little more challenging. In this section,
we will provide short examples of data preparation and modeling for
multivariate multi-step
time series forecasting as a template to ease this challenge,
specifically:

1.  Multiple Input Multi-step Output.
2.  Multiple Parallel Input and Multi-step Output.

<!-- -->

Perhaps the biggest stumbling block is in the preparation of data, so this is where we will
focus our attention.

#### Multiple Input Multi-step Output

There are those multivariate time series forecasting problems where the
output series is separate
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

```

```
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
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `15_multivariate_dependent_series_multi_step_dataset.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first prints the shape of the prepared training data. We can see that
the shape of the input portion of the samples is three-dimensional, comprised of six samples,
with three time steps, and two variables for the 2 input time series. The output portion of the
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

We can now develop an LSTM model for multi-step predictions. A vector output or an
encoder-decoder model could be used. In this case, we will demonstrate a vector output with a
Stacked LSTM. The complete example is listed below.

```
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
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

n_features = X.shape[2]

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True,
input_shape=(n_steps_in,
n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

##### Run Notebook
Click notebook `16_stacked_lstm_dependent_multi_step.ipynb` in jupterLab UI and run jupyter notebook.

Running the example fits the model and predicts the next two time steps of the output
sequence beyond the dataset. We would expect the next two steps to be:[185, 205]. It is a
challenging framing of the problem with very little data, and the arbitrarily configured version
of the model gets close.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[188.70619 210.16513]]

```

#### Multiple Parallel Input and Multi-step Output

A problem with parallel time series may require the prediction of multiple time steps of each
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

```


```
Output:
40, 45, 85
50, 55, 105

```

The splitsequences() function below implements this behavior.


```
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

```


We can demonstrate this function on the small contrived dataset. The
complete example is listed below.

```
# multivariate multi-step data preparation
from numpy import array
from numpy import hstack

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
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

```

##### Run Notebook
Click notebook `17_multivariate_parallel_series_multi_step_dataset.ipynb` in jupterLab UI and run jupyter notebook.


Running the example first prints the shape of the prepared training dataset. We can see
that both the input (X) and output (Y) elements of the dataset are three
dimensional for the
number of samples, time steps, and variables or parallel time series
respectively. The input and
output elements of each series are then printed side by side so that we
can confirm that the
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

We can use either the Vector Output or Encoder-Decoder LSTM to model
this problem. In
this case, we will use the Encoder-Decoder model. The complete example
is listed below.

```
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

def split_sequences(sequences, n_steps_in, n_steps_out):
X, y = list(), list()
for i in range(len(sequences)):


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

n_features = X.shape[2]

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in,
n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=300, verbose=0)

x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `18_encoder_decoder_lstm_parallel_multi_step.ipynb` in jupterLab UI and run jupyter notebook.

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
[[[ 91.86044 97.77231 189.66768 ]

[103.299355 109.18123 212.6863 ]]]

```

## Exercises

This section lists some ideas for extending the tutorial that you may
wish to explore.

- Problem Differences. Explain the main changes to the LSTM required when modeling
each of univariate, multivariate and multi-step time series forecasting problems.

- Experiment. Select one example and modify it to work with your own small contrived
dataset.

- Develop Framework. Use the examples in this chapter as the basis for a framework for
automatically developing an LSTM model for a given time series forecasting problem.


### Summary

In this tutorial, you discovered how to develop a suite of LSTM models for a range of standard
time series forecasting problems. Specifically, you learned:

- How to develop LSTM models for univariate time series forecasting.

- How to develop LSTM models for multivariate time series forecasting.

- How to develop LSTM models for multi-step time series forecasting.
