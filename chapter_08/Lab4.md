<img align="right" src="../logo-small.png">


### How to Develop CNNs for Time Series Forecasting (Part 1)

Convolutional Neural Network models, or CNNs for short, can be applied to time series
forecasting. There are many types of CNN models that can be used for each specific type of
time series forecasting problem. In this tutorial, you will discover how to develop a suite of
CNN models for a range of standard time series forecasting problems. The objective of this
tutorial is to provide standalone examples of each model on each type of time series problem as
a template that you can copy and adapt for your specific time series forecasting problem. After
completing this tutorial, you will know:

- How to develop CNN models for univariate time series forecasting.
- How to develop CNN models for multivariate time series forecasting.
- How to develop CNN models for multi-step time series forecasting.

Let' s get started.

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-time-series-forecasting` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab4_CNNs_Time_Series_Forecasting_Part1`

#### Tutorial Overview
In this tutorial, we will explore how to develop CNN models for time series forecasting. The
models are demonstrated on small contrived time series problems intended to give the flavor
of the type of time series problem being addressed. The chosen configuration of the models is
arbitrary and not optimized for each problem; that was not the goal. This tutorial is divided
into four parts; they are:

1. Univariate CNN Models
2. Multivariate CNN Models
3. Multi-step CNN Models
4. Multivariate Multi-step CNN Models

#### Univariate CNN Models
Although traditionally developed for two-dimensional image data, CNNs can be used to model
univariate time series forecasting problems. Univariate time series are datasets comprised of a
single series of observations with a temporal ordering and a model is required to learn from the
series of past observations to predict the next value in the sequence. This section is divided into
two parts; they are:

1. Data Preparation
2. CNN Model

#### Data Preparation

Before a univariate series can be modeled, it must be prepared. The CNN model will learn a
function that maps a sequence of past observations as input to an output observation. As such,
the sequence of observations must be transformed into multiple examples from which the model
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

# split a univariate sequence into samples
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

```


We can demonstrate this function on our small contrived dataset above. The complete
example is listed below.

```
# univariate data preparation
from numpy import array

# split a univariate sequence into samples
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
# summarize the data
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `01_univariate_dataset.ipynb` in jupterLab UI and run jupyter notebook.

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

Now that we know how to prepare a univariate series for modeling, let's look at developing
a CNN model that can learn the mapping of inputs to outputs.

#### CNN Model

A one-dimensional CNN is a CNN model that has a convolutional hidden
layer that operates
over a 1D sequence. This is followed by perhaps a second convolutional layer in some cases,
such as very long input sequences, and then a pooling layer whose job it is to distill the output
of the convolutional layer to the most salient elements. The convolutional and pooling layers
are followed by a dense fully connected layer that interprets the features extracted by the
convolutional part of the model. A flatten layer is used between the convolutional layers and
the dense layer to reduce the feature maps to a single one-dimensional vector. We can define a
1D CNN Model for univariate time series forecasting as follows.

```
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps,
n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

```

Key in the definition is the shape of the input; that is what the model expects as input for
each sample in terms of the number of time steps and the number of features. We are working
with a univariate series, so the number of features is one, for one
variable. The number of
time steps as input is the number we chose when preparing our dataset as an argument to the
splitsequence() function.

The input shape for each sample is specified in the input shape argument on the definition
of the first hidden layer. We almost always have multiple samples, therefore, the model will
expect the input component of training data to have the dimensions or shape: [samples,
timesteps, features]. Oursplitsequence() function in the previous section outputs the
X with the shape[samples, timesteps], so we can easily reshape it to have an additional
dimension for the one feature.

```
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

```

The CNN does not actually view the data as having time steps, instead, it is treated as a
sequence over which convolutional read operations can be performed, like a one-dimensional
image. In this example, we define a convolutional layer with 64 filter maps and a kernel size
of 2. This is followed by a max pooling layer and a dense layer to interpret the input feature.

An output layer is specified that predicts a single numerical value. The
model is fit using the
efficient Adam version of stochastic gradient descent and optimized using the mean squared
error, or 'mse' , loss function. Once the model is defined, we can fit it on the training dataset.

```

# fit model
model.fit(X, y, epochs=1000, verbose=0)

```
After the model is fit, we can use it to make a prediction. We can predict the next value
in the sequence by providing the input:[70, 80, 90]. And expecting the model to predict
something like: [100]. The model expects the input shape to be three-dimensional with
[samples, timesteps, features], therefore, we must reshape the single input sample before
making the prediction.

```
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

```

We can tie all of this together and demonstrate how to develop a 1D CNN model for
univariate time series forecasting and make a single prediction.

```
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
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

n_steps = 3

X, y = split_sequence(raw_seq, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
input_shape=(n_steps,
n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=1000, verbose=0)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))


yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `02_cnn_univariate.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model, and makes a prediction. We can see
that the model predicts the next value in the sequence.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
[[101.67965]]

```

For an example of a CNN applied to a real-world univariate time series forecasting problem
see Chapter 14. For an example of grid searching CNN hyperparameters on a univariate time
series forecasting problem, see Chapter 15.

### Multivariate CNN Models

Multivariate time series data means data where there is more than one observation for each
time step. There are two main models that we may require with multivariate time series data;
they are:

1.  Multiple Input Series.
2.  Multiple Parallel Series.

Let' s take a look at each in turn.

#### Multiple Input Series

A problem may have two or more parallel input time series and an output
time series that is
dependent on the input time series. The input time series are parallel because each series has
observations at the same time steps. We can demonstrate this with a simple example of two
parallel input time series where the output series is the simple addition of the input series.

```

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

```

We can reshape these three arrays of data as a single dataset where each row is a time step
and each column is a separate time series. This is a standard way of storing parallel time series
in a CSV file.

```

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))

```

The complete example is listed below.

```

from numpy import array
from numpy import hstack

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in
range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)

```

##### Run Notebook
Click notebook `03_dependent_time_series_dataset.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prints the dataset with one row per time step and
one column for each
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
and output samples. A 1D CNN model needs sufficient context to learn a
mapping from an
input sequence to an output value. CNNs can support parallel input time
series as separate
channels, like red, green, and blue components of an image. Therefore,
we need to split the
data into samples maintaining the order of observations across the two
input sequences. If we
chose three input time steps, then the first sample would look as
follows:

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

That is, the first three time steps of each parallel series are provided
as input to the model
and the model associates this with the value in the output series at the
third time step, in this
case, 65. We can see that, in transforming the time series into
input/output samples to train
the model, that we will have to discard some values from the output time
series where we do
not have values in the input time series at prior time steps. In turn,
the choice of the size of
the number of input time steps will have an important effect on how much
of the training data
is used. We can define a function namedsplitsequences() that will take a
dataset as we
have defined it with rows for time steps and columns for parallel series
and return input/outputsamples.

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
input time series as input. The complete example is listed below.

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
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
print(X[i], y[i])

```

##### Run Notebook
Click notebook `04_split_samples_dependent_time_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first prints the shape of the `X` and `y` components. We can see that the
Xcomponent has a three-dimensional structure. The first dimension is the number of samples,
in this case 7. The second dimension is the number of time steps per sample, in this case 3, the
value specified to the function. Finally, the last dimension specifies
the number of parallel time
series or the number of variables, in this case 2 for the two parallel series.
This is the exact three-dimensional structure expected by a 1D CNN as input. The data
is ready to use without further reshaping. We can then see that the input and output for
each sample is printed, showing the three time steps for each of the two input series and the
associated output for each sample.

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


#### CNN Model

We are now ready to fit a 1D CNN model on this data, specifying the
expected number of time
steps and features to expect for each input sample, in this case three and two respectively.

```
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps,
n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
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
be [1, 3, 2].
We would expect the next value in the sequence to be 100 + 105 or 205.

```

# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

```

The complete example is listed below.

```

# multivariate cnn example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps,
n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `05_cnn_multivariate_dependent_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model, and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```

[[206.0161]]

```

#### Multi-headed CNN Model

There is another, more elaborate way to model the problem. Each input
series can be handled by
a separate CNN and the output of each of these submodels can be combined before a prediction
is made for the output sequence. We can refer to this as a multi-headed CNN model. It may
offer more flexibility or better performance depending on the specifics of the problem that is
being modeled. For example, it allows you to configure each submodel
differently for each input
series, such as the number of filter maps and the kernel size. This type
of model can be defined
in Keras using the Keras functional API. First, we can define the first
input model as a 1D
CNN with an input layer that expects vectors withnstepsand 1 feature.

```

visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)

```
We can define the second input submodel in the same way.

```

visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
cnn2 = MaxPooling1D(pool_size=2)(cnn2)
cnn2 = Flatten()(cnn2)

```

Now that both input submodels have been defined, we can merge the output from each
model into one long vector which can be interpreted before making a
prediction for the output sequence.

```

merge = concatenate([cnn1, cnn2])
dense = Dense(50, activation='relu')(merge)
output = Dense(1)(dense)

```
We can then tie the inputs and outputs together.

```

model = Model(inputs=[visible1, visible2], outputs=output)

```

The image below provides a schematic for how this model looks, including
the shape of the inputs and outputs of each layer.

![](./images/116-4.png)

This model requires input to be provided as a list of two elements where
each element in the
list contains data for one of the submodels. In order to achieve this,
we can split the 3D input
data into two separate arrays of input data; that is from one array with
the shape[7, 3, 2]
to two 3D arrays with [7, 3, 1].

```

n_features = 1

X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)

```

These data can then be provided in order to fit the model.

```
model.fit([X1, X2], y, epochs=1000, verbose=0)

```

Similarly, we must prepare the data for a single sample as two separate two-dimensional
arrays when making a single one-step prediction.

```

x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
x2 = x_input[:, 1].reshape((1, n_steps, n_features))

```


We can tie all of this together; the complete example is listed below.

```

from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

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

n_features = 1

X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)

visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)

visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
cnn2 = MaxPooling1D(pool_size=2)(cnn2)


cnn2 = Flatten()(cnn2)
# merge input models
merge = concatenate([cnn1, cnn2])
dense = Dense(50, activation='relu')(merge)
output = Dense(1)(dense)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit([X1, X2], y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
x2 = x_input[:, 1].reshape((1, n_steps, n_features))
yhat = model.predict([x1, x2], verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `06_multiheaded_cnn_multivariate_dependent_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model, and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```

[[205.871]]

```

For an example of CNN models developed for a multivariate time series classification problem,
see Chapter 24.

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
Click notebook `07_split_parallel_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example first prints the shape of the prepared `X` and `y` components. The
shape ofXis three-dimensional, including the number of samples (6), the number of time steps
chosen per sample (3), and the number of parallel time series or features (3). The shape ofy
is two-dimensional as we might expect for the number of samples (6) and the number of time
variables per sample to be predicted (3). The data is ready to use in a
1D CNN model that
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

#### Vector-Output CNN Model

We are now ready to fit a 1D CNN model on this data. In this model, the
number of time steps
and parallel series (features) are specified for the input layer via
the input shape argument.
The number of parallel series is also used in the specification of the
number of values to predict
by the model in the output layer; again, this is three.

```

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps,
n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

```

We can predict the next value in each of the three parallel series by providing an input of
three time steps for each series.

```
70, 75, 145
80, 85, 165
90, 95, 185

```

The shape of the input for making a single prediction must be 1 sample, 3 time steps, and 3
features, or[1, 3, 3].

```

# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

```
We would expect the vector output to be: [100, 105, 205]. We can tie all of this together
and demonstrate a 1D CNN for multivariate output time series forecasting below.

```

# multivariate output 1d cnn example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps,
n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=3000, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

```

##### Run Notebook
Click notebook `08_cnn_multivariate_parallel_series.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prepares the data, fits the model and makes a
prediction.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```

[[100.11272 105.32213 205.53436]]

```

#### Next
In the next lab, we will work on part 2 of time series forecasting.
