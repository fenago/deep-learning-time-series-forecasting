<img align="right" src="../logo-small.png">


### How to Develop LSTMs for Time Series Forecasting

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

Let’s get started.

### Tutorial Overview

In this tutorial, we will explore how to develop a suite of different types of LSTM models for
time series forecasting. The models are demonstrated on small contrived time series problems
intended to give the flavor of the type of time series problem being addressed. The chosen
configuration of the models is arbitrary and not optimized for each problem; that was not the
goal. This tutorial is divided into four parts; they are:

1.  Univariate LSTM Models
2.  Multivariate LSTM Models
3.  Multi-step LSTM Models
4.  Multivariate Multi-step LSTM Models

### Univariate LSTM Models

LSTMs can be used to model univariate time series forecasting problems. These are problems
comprised of a single series of observations and a model is required to learn from the series of
past observations to predict the next value in the sequence. We will demonstrate a number of

variations of the LSTM model for univariate time series forecasting.
This section is divided into

six parts; they are:

1.  Data Preparation
2.  Vanilla LSTM
3.  Stacked LSTM
4.  Bidirectional LSTM
5.  CNN-LSTM
6.  ConvLSTM

<!-- -->

Each of these models are demonstrated for one-step univariate time series forecasting, but
can easily be adapted and used as the input part of a model for other types of time series
forecasting problems.

#### Data Preparation

Before a univariate series can be modeled, it must be prepared. The LSTM model will learn a
function that maps a sequence of past observations as input to an output observation. As such,
the sequence of observations must be transformed into multiple examples from which the LSTM
can learn. Consider a given univariate sequence:

```
[10, 20, 30, 40, 50, 60, 70, 80, 90]

```

We can divide the sequence into multiple input/output patterns called samples, where three
time steps are used as input and one time step is used as output for the one-step prediction
that is being learned.

```
X, y
10, 20, 30, 40
20, 30, 40, 50
30, 40, 50, 60
...

```

The splitsequence() function below implements this behavior and will split a given
univariate sequence into multiple samples where each sample has a specified number of time
steps and the output is a single time step.

```
def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):

end_ix = i + n_steps

if end_ix > len(sequence)-1:
break

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

```

We can demonstrate this function on our small contrived dataset above. The complete
example is listed below.

```
from numpy import array

def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):

end_ix = i + n_steps

if end_ix > len(sequence)-1:
break

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 3

X, y = split_sequence(raw_seq, n_steps)

for i in range(len(X)):
print(X[i], y[i])

```

Running the example splits the univariate series into six samples where each sample has
three input time steps and one output time step.

```
[10 20 30] 40
[20 30 40] 50
[30 40 50] 60

[40 50 60] 70

[50 60 70] 80

[60 70 80] 90

```

Now that we know how to prepare a univariate series for modeling, let’s look at developing
LSTM models that can learn the mapping of inputs to outputs, starting with a Vanilla LSTM.

#### Vanilla LSTM

A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM
units, and an
output layer used to make a prediction. Key to LSTMs is that they offer native support for
sequences. Unlike a CNN that reads across the entire input vector, the LSTM model reads one
time step of the sequence at a time and builds up an internal state representation that can be
used as a learned context for making a prediction. We can define a Vanilla LSTM for univariate
time series forecasting as follows.

```
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

```

Key in the definition is the shape of the input; that is what the model expects as input for
each sample in terms of the number of time steps and the number of features. We are working
with a univariate series, so the number of features is one, for one
variable. The number of
time steps as input is the number we chose when preparing our dataset as an argument to the
splitsequence() function.

The shape of the input for each sample is specified in the input shape argument on the
definition of first hidden layer. We almost always have multiple samples, therefore, the model
will expect the input component of training data to have the dimensions
or shape:[samples,
timesteps, features]. Oursplitsequence() function in the previous section
outputs theX
with the shape[samples, timesteps], so we easily reshape it to have an
additional dimension
for the one feature.

```
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

```
In this case, we define a model with 50 LSTM units in the hidden layer and an output layer
that predicts a single numerical value. The model is fit using the efficient Adam version of
stochastic gradient descent and optimized using the mean squared error, or‘mse’loss function.
Once the model is defined, we can fit it on the training dataset.

```
# fit model
model.fit(X, y, epochs=200, verbose=0)

```

After the model is fit, we can use it to make a prediction. We can predict the next value
in the sequence by providing the input:[70, 80, 90]. And expecting the
model to predict
something like: [100]. The model expects the input shape to be
three-dimensional with
[samples, timesteps, features], therefore, we must reshape the single
input sample before
making the prediction.

```
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

```

We can tie all of this together and demonstrate how to develop a Vanilla
LSTM for univariate
time series forecasting and make a single prediction.

```
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):

end_ix = i + n_steps

if end_ix > len(sequence)-1:
break

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 3

X, y = split_sequence(raw_seq, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps,
n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))


yhat = model.predict(x_input, verbose=0)
print(yhat)

```

Running the example prepares the data, fits the model, and makes a prediction. We can see
that the model predicts the next value in the sequence.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[102.09213]]

```

#### Stacked LSTM

Multiple hidden LSTM layers can be stacked one on top of another in what is referred to as
a Stacked LSTM model. An LSTM layer requires a three-dimensional input and LSTMs by
default will produce a two-dimensional output as an interpretation from the end of the sequence.

We can address this by having the LSTM output a value for each time step
in the input data by
setting thereturnsequences=Trueargument on the layer. This allows us to have 3D output
from hidden LSTM layer as input to the next. We can therefore define a Stacked LSTM as
follows.

```
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps,
n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

```
We can tie this together; the complete code example is listed below.

```
# univariate stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence
def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the sequence
if end_ix > len(sequence)-1:
break
# gather input and output parts of the pattern
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]


X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps,
n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

Running the example predicts the next value in the sequence, which we
expect would be 100.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[102.47341]]

```

#### Bidirectional LSTM

On some sequence prediction problems, it can be beneficial to allow the LSTM model to learn
the input sequence both forward and backwards and concatenate both interpretations. This
is called a Bidirectional LSTM. We can implement a Bidirectional LSTM for univariate time
series forecasting by wrapping the first hidden layer in a wrapper layer called Bidirectional.

An example of defining a Bidirectional LSTM to read input both forward
and backward is as follows.

```
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

```

The complete example of the Bidirectional LSTM for univariate time series forecasting is listed below.

```
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):

end_ix = i + n_steps

if end_ix > len(sequence)-1:
break

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 3

X, y = split_sequence(raw_seq, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'),
input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

Running the example predicts the next value in the sequence, which we
expect would be 100.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider
running the example a few times.

```
[[101.48093]]

```


#### CNN-LSTM

A convolutional neural network, or CNN for short, is a type of neural
network developed for
working with two-dimensional image data. The CNN can be very effective
at automatically
extracting and learning features from one-dimensional sequence data such as univariate time
series data. A CNN model can be used in a hybrid model with an LSTM backend where the
CNN is used to interpret subsequences of input that together are provided as a sequence to an
LSTM model to interpret. This hybrid model is called a CNN-LSTM.
The first step is to split the input sequences into subsequences that can be processed by the
CNN model. For example, we can first split our univariate time series data into input/output
samples with four steps as input and one as output. Each sample can then be split into two
sub-samples, each with two time steps. The CNN can interpret each subsequence of two time
steps and provide a time series of interpretations of the subsequences to the LSTM model
to process as input. We can parameterize this and define the number of subsequences as
nseqand the number of time steps per subsequence asnsteps. The input data can then be
reshaped to have the required structure:[samples, subsequences, timesteps, features].
For example:

```
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

```

We want to reuse the same CNN model when reading in each sub-sequence of
data separately.

This can be achieved by wrapping the entire CNN model in
aTimeDistributedwrapper that
will apply the entire model once per input, in this case, once per input
subsequence. The CNN
model first has a convolutional layer for reading across the subsequence that requires a number
of filters and a kernel size to be specified. The number of filters is the number of reads or
interpretations of the input sequence. The kernel size is the number of time steps included of
eachreadoperation of the input sequence. The convolution layer is followed by a max pooling
layer that distills the filter maps down to^14 of their size that includes the most salient features.

These structures are then flattened down to a single one-dimensional
vector to be used as a
single input time step to the LSTM layer.

```
# define the input cnn model
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))

```

Next, we can define the LSTM part of the model that interprets the CNN model’s read of
the input sequence and makes a prediction.

```
# define the output model

model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

```

We can tie all of this together; the complete example of a CNN-LSTM
model for univariate
time series forecasting is listed below.

```
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):

end_ix = i + n_steps

if end_ix > len(sequence)-1:
break

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 4

X, y = split_sequence(raw_seq, n_steps)

n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,
activation='relu'),
input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=500, verbose=0)

x_input = array([60, 70, 80, 90])


x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

Running the example predicts the next value in the sequence, which we
expect would be 100.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[101.69263]]

```

#### ConvLSTM

A type of LSTM related to the CNN-LSTM is the ConvLSTM, where the
convolutional reading
of input is built directly into each LSTM unit. The ConvLSTM was developed for reading
two-dimensional spatial-temporal data, but can be adapted for use with univariate time series
forecasting. The layer expects input as a sequence of two-dimensional images, therefore the
shape of input data must be:[samples, timesteps, rows, columns, features].
For our purposes, we can split each sample into subsequences where timesteps will become
the number of subsequences, ornseq, and columns will be the number of time steps for
each subsequence, ornsteps. The number of rows is fixed at 1 as we are working with
one-dimensional data. We can now reshape the prepared samples into the required structure.

```
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

```
We can define the ConvLSTM as a single layer in terms of the number of filters and a two-
dimensional kernel size in terms of(rows, columns). As we are working with a one-dimensional
series, the number of rows is always fixed to 1 in the kernel. The output of the model must
then be flattened before it can be interpreted and a prediction made.

```
# define the input cnnlstm model
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq,
1, n_steps, n_features)))
model.add(Flatten())

```

The complete example of a ConvLSTM for one-step univariate time series forecasting is
listed below.


```
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):

end_ix = i + n_steps

if end_ix > len(sequence)-1:
break

seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 4

X, y = split_sequence(raw_seq, n_steps)

n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu',
input_shape=(n_seq,
1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=500, verbose=0)

x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

Running the example predicts the next value in the sequence, which we
expect would be 100.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.


```
[[103.68166]]

```

For an example of an LSTM applied to a real-world univariate time series forecasting problem
see Chapter 14. For an example of grid searching LSTM hyperparameters on a univariate time
series forecasting problem, see Chapter 15. Now that we have looked at LSTM models for
univariate data, let’s turn our attention to multivariate data.

### Multivariate LSTM Models

Multivariate time series data means data where there is more than one observation for each
time step. There are two main models that we may require with multivariate time series data;
they are:

1.  Multiple Input Series.
2.  Multiple Parallel Series.

Let’s take a look at each in turn.

#### Multiple Input Series

A problem may have two or more parallel input time series and an output
time series that is

dependent on the input time series. The input time series are parallel because each series has
an observation at the same time steps. We can demonstrate this with a simple example of two
parallel input time series where the output series is the simple addition of the input series.

```
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

```

We can reshape these three arrays of data as a single dataset where each row is a time step,
and each column is a separate time series. This is a standard way of storing parallel time series in a CSV file.

```
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

```

The complete example is listed below.


```
# multivariate data preparation
from numpy import array
from numpy import hstack
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
print(dataset)

```
Running the example prints the dataset with one row per time step and one column for each
of the two input and one output parallel time series.

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
As with the univariate time series, we must structure these data into samples with input
and output elements. An LSTM model needs sufficient context to learn a mapping from an
input sequence to an output value. LSTMs can support parallel input time series as separate
variables or features. Therefore, we need to split the data into samples
maintaining the order of
observations across the two input sequences. If we chose three input time steps, then the first
sample would look as follows:


```
Input:

10, 15
20, 25
30, 35

```


```
Output:

65

```
That is, the first three time steps of each parallel series are provided as input to the model
and the model associates this with the value in the output series at the third time step, in this
case, 65. We can see that, in transforming the time series into input/output samples to train
the model, that we will have to discard some values from the output time series where we do
not have values in the input time series at prior time steps. In turn,
the choice of the size of
the number of input time steps will have an important effect on how much
of the training data
is used. We can define a function namedsplitsequences() that will take a
dataset as we
have defined it with rows for time steps and columns for parallel series
and return input/output
samples.

```
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):

end_ix = i + n_steps

if end_ix > len(sequences):
break

seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
X.append(seq_x)
y.append(seq_y)
return array(X), array(y)

```

We can test this function on our dataset using three time steps for each
input time series as
input. The complete example is listed below.

```
from numpy import array
from numpy import hstack

def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):

end_ix = i + n_steps

if end_ix > len(sequences):
break

seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
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
print(X.shape, y.shape)

for i in range(len(X)):
print(X[i], y[i])

```

Running the example first prints the shape of the `X` and `y` components. We can
see that
the `X` component has a three-dimensional structure. The first dimension is
the number of
samples, in this case 7. The second dimension is the number of time
steps per sample, in this
case 3, the value specified to the function. Finally, the last dimension
specifies the number of
parallel time series or the number of variables, in this case 2 for the
two parallel series. This is
the exact three-dimensional structure expected by an LSTM as input. The
data is ready to
use without further reshaping. We can then see that the input and output
for each sample is
printed, showing the three time steps for each of the two input series
and the associated output
for each sample.

```
(7, 3, 2) (7,)

[[10 15]
[20 25]
[30 35]] 65
[[20 25]
[30 35]
[40 45]] 85
[[30 35]
[40 45]
[50 55]] 105
[[40 45]
[50 55]
[60 65]] 125
[[50 55]
[60 65]
[70 75]] 145
[[60 65]
[70 75]
[80 85]] 165
[[70 75]
[80 85]
[90 95]] 185

```

We are now ready to fit an LSTM model on this data. Any of the varieties
of LSTMs in the
previous section can be used, such as a Vanilla, Stacked, Bidirectional,
CNN, or ConvLSTM
model. We will use a Vanilla LSTM where the number of time steps and
parallel series (features)
are specified for the input layer via the input shape argument.

```
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps,
n_features)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

```

When making a prediction, the model expects three time steps for two input time series.
We can predict the next value in the output series providing the input
values of:

```
80, 85
90, 95
100, 105

```

The shape of the one sample with three time steps and two variables must
be [1, 3, 2]. We would expect the next value in the sequence to be 100 + 105, or 205.

```
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

```
The complete example is listed below.

```
# multivariate lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
X, y = list(), list()
for i in range(len(sequences)):
# find the end of this pattern
end_ix = i + n_steps
# check if we are beyond the dataset
if end_ix > len(sequences):
break
# gather input and output parts of the pattern
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
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
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

Running the example prepares the data, fits the model, and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[208.13531]]

```

For an example of LSTM models developed for a multivariate time series classification
problem, see Chapter 25.

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

Before we look at these models, let’s first look at the preparation of data for multi-step
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

Running the example splits the univariate series into input and output
time steps and prints the input and output components of each.

```
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]

```

Now that we know how to prepare data for multi-step forecasting, let’s
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
expects data to have a three-dimensional structure of[samples,
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
output of the encoder is a fixed length vector that represents the model’s interpretation of the
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
dimensional shape of[samples, timesteps, features].

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

Running the example forecasts and prints the next two time steps in the
sequence.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.


```
[[[101.9736

[116.213615]]]

```

For an example of LSTM models developed for a multi-step time series forecasting problem,
see Chapter 20.

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

For an example of LSTM models developed for a multivariate multi-step time series forecasting
problem, see Chapter 20.

### Extensions

This section lists some ideas for extending the tutorial that you may
wish to explore.

- Problem Differences. Explain the main changes to the LSTM required when modeling
each of univariate, multivariate and multi-step time series forecasting problems.

- Experiment. Select one example and modify it to work with your own small contrived
dataset.

- Develop Framework. Use the examples in this chapter as the basis for a framework for
automatically developing an LSTM model for a given time series forecasting problem.


### Further Reading

This section provides more resources on the topic if you are looking to
go deeper.

#### Papers

- Long Short-Term Memory, 1997.
https://ieeexplore.ieee.org/document/6795963/.

- Learning to Forget: Continual Prediction with LSTM, 1999.
https://ieeexplore.ieee.org/document/818041/

- Recurrent Nets that Time and Count, 2000.
https://ieeexplore.ieee.org/document/861302/

- LSTM: A Search Space Odyssey, 2017.
https://arxiv.org/abs/1503.04069

- Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting,
2015. https://arxiv.org/abs/1506.04214v1

#### APIs

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

- Keras Recurrent Layers API.
https://keras.io/layers/recurrent/

### Summary

In this tutorial, you discovered how to develop a suite of LSTM models for a range of standard
time series forecasting problems. Specifically, you learned:

- How to develop LSTM models for univariate time series forecasting.

- How to develop LSTM models for multivariate time series forecasting.

- How to develop LSTM models for multi-step time series forecasting.
