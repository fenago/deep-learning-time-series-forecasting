<img align="right" src="../logo-small.png">


### How to Prepare Time Series Data for CNNs and LSTMs

Time series data must be transformed before it can be used to fit a
supervised learning model.
In this form, the data can be used immediately to fit a supervised machine learning algorithm
and even a Multilayer Perceptron neural network. One further transformation is required in
order to ready the data for fitting a Convolutional Neural Network (CNN) or Long Short-Term
Memory (LSTM) Neural Network. Specifically, the two-dimensional structure of the supervised
learning data must be transformed to a three-dimensional structure. This is perhaps the largest
sticking point for practitioners looking to implement deep learning methods for time series
forecasting. In this tutorial, you will discover exactly how to transform a time series data set
into a three-dimensional structure ready for fitting a CNN or LSTM model. After completing
this tutorial, you will know:

- How to transform a time series dataset into a two-dimensional
supervised learning format.

- How to transform a two-dimensional time series dataset into a three-dimensional structure
suitable for CNNs and LSTMs.

- How to step through a worked example of splitting a very long time series into subsequences
ready for training a CNN or LSTM model.

Let’s get started.

### Overview

This tutorial is divided into four parts, they are:

1.  Time Series to Supervised.
2.  3D Data Preparation Basics.
3.  Univariate Worked Example.

### Time Series to Supervised

Time series data requires preparation before it can be used to train a
supervised learning model,

such as an LSTM neural network. For example, a univariate time series is represented as a

vector of observations:

```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

A supervised learning algorithm requires that data is provided as a collection of samples,

where each sample has an input component (X) and an output component
(y).

```
X, y
sample input, sample output
sample input, sample output
sample input, sample output
...
```

The model will learn how to map inputs to outputs from the provided
examples.

y=f(X) (6.1)

A time series must be transformed into samples with input and output components. The
transform both informs what the model will learn and how you intend to use the model in
the future when making predictions, e.g. what is required to make a prediction (X) and what
prediction is made (y). For a univariate time series problem where we are interested in one-step
predictions, the observations at prior time steps, so-called lag observations, are used as input
and the output is the observation at the current time step. For example, the above 10-step
univariate series can be expressed as a supervised learning problem with three time steps for
input and one step as output, as follows:

```
X, y
[1, 2, 3], [4]
[2, 3, 4], [5]
[3, 4, 5], [6]
...
```

For more on transforming your time series data into a supervised learning problem in general
see Chapter 4. You can write code to perform this transform yourself and that is the general
approach I teach and recommend for greater understanding of your data and control over the
transformation process. The splitsequence() function below implements this behavior and

will split a given univariate sequence into multiple samples where each
sample has a specified

number of time steps and the output is a single time step.

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

For specific examples for univariate, multivariate and multi-step time series, see Chapters 7,
8 and 9. After you have transformed your data into a form suitable for training a supervised
learning model it will be represented as rows and columns. Each column will represent a feature
to the model and may correspond to a separate lag observation. Each row will represent a
sample and will correspond to a new example with input and output components.

- Feature: A column in a dataset, such as a lag observation for a time
series dataset.

- Sample: A row in a dataset, such as an input and output sequence for a time series
dataset.

For example, our univariate time series may look as follows:

```
x1, x2, x3, y
1, 2, 3, 4
2, 3, 4, 5
3, 4, 5, 6
...

```

The dataset will be represented in Python using a NumPy array. The array will have two
dimensions. The length of each dimension is referred to as the shape of the array. For example,
a time series with 3 inputs, 1 output will be transformed into a supervised learning problem

with 4 columns, or really 3 columns for the input data and 1 for the
output data. If we have 7

rows and 3 columns for the input data then the shape of the dataset would be [7, 3], or 7
samples and 3 features. We can make this concrete by transforming our small contrived dataset.

```
# transform univariate time series to supervised learning problem
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
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(series.shape)
X, y = split_sequence(series, 3)
print(X.shape, y.shape)
for i in range(len(X)):
print(X[i], y[i])
```

Running the example first prints the shape of the time series, in this case 10 time steps

of observations. Next, the series is split into input and output
components for a supervised

learning problem. We can see that for the chosen representation that we
have 7 samples for the

input and output and 3 input features. The shape of the output is 7
samples represented as(7,)

indicating that the array is a single column. It could also be
represented as a two-dimensional

array with 7 rows and 1 column[7, 1]. Finally , the input and output
aspects of each sample

are printed, showing the expected breakdown of the problem.

```
(10,)

(7, 3) (7,)

[1 2 3] 4
[2 3 4] 5
[3 4 5] 6
[4 5 6] 7
[5 6 7] 8
[6 7 8] 9
[7 8 9] 10

```

Data in this form can be used directly to train a simple neural network,
such as a Multilayer

Perceptron. The difficulty for beginners comes when trying to prepare
this data for CNNs and

LSTMs that require data to have a three-dimensional structure instead of
the two-dimensional

structure described so far.

### Data Preparation Basics

Preparing time series data for CNNs and LSTMs requires one additional
step beyond transforming

the data into a supervised learning problem. This one additional step
causes the most confusion

for beginners. In this section we will slowly step through the basics of
how and why we need to

prepare three-dimensional data for CNNs and LSTMs before working through
an example in

the next section.

The input layer for CNN and LSTM models is specified by
the input shape argument on

the first hidden layer of the network. This too can make things
confusing for beginners as

intuitively we may expect the first layer defined in the model be the
input layer, not the first
hidden layer. For example, below is an example of a network with one
hiddenLSTMlayer and

oneDenseoutput layer.

```
# lstm without an input layer
...
model = Sequential()
model.add(LSTM(32))
model.add(Dense(1))

```

In this example, theLSTM()layer must specify the shape of the input
data. The input to

every CNN and LSTM layer must be three-dimensional. The three dimensions
of this input are:

- Samples. One sequence is one sample. A batch is comprised of one or
more samples.

- Time Steps. One time step is one point of observation in the sample. One sample is
comprised of multiple time steps.

- Features. One feature is one observation at a time step. One time step is comprised of
one or more features.

This expected three-dimensional structure of input data is often
summarized using the array

shape notation of:[samples, timesteps, features]. Remember, that the
two-dimensional

shape of a dataset that we are familiar with from the previous section
has the array shape of:

[samples, features]. this means we are adding the new dimension oftime
steps. Except, in

time series forecasting problems our features are observations at time
steps. So, really, we are

adding the dimension offeatures, where a univariate time series has only
one feature.

When defining the input layer of your LSTM network, the network assumes you have one

or more samples and requires that you specify the number of time steps
and the number of

features. You can do this by specifying a tuple to
the input shape argument. For example, the

model below defines an input layer that expects 1 or more samples, 3
time steps, and 1 feature.

Remember, the first layer in the network is actually the first hidden
layer, so in this example 32

refers to the number of units in the first hidden layer. The number of
units in the first hidden

layer is completely unrelated to the number of samples, time steps or
features in your input

data.

```
# lstm with an input layer

...
model = Sequential()
model.add(LSTM(32, input_shape=(3, 1)))
model.add(Dense(1))

```

This example maps onto our univariate time series from the previous
section that we split into

having 3 input time steps and 1 feature. We may have loaded our time
series dataset from CSV

or transformed it to a supervised learning problem in memory. It will
have a two-dimensional

shape and we must convert it to a three-dimensional shape with some
number of samples, 3

time steps per sample and 1 feature per time step, or[?, 3, 1]. We can
do this by using the

reshape()NumPy function. For example, if we have 7 samples and 3 time
steps per sample for

the input element of our time series, we can reshape it into[7, 3, 1]by
providing a tuple to


thereshape() function specifying the desired new shape of(7, 3, 1). The
array must have

enough data to support the new shape, which in this case it does as[7,
3]and[7, 3, 1]are

functionally the same thing.

```
...
X = X.reshape((7, 3, 1))

```

A short-cut in reshaping the array is to use the known shapes, such as
the number of samples

and the number of times steps from the array returned from the call to
theX.shapeproperty of

the array. For example,X.shape[0]refers to the number of rows in a 2D
array, in this case the

number of samples andX.shape[1]refers to the number of columns in a 2D
array, in this case

the number of feature that we will use as the number of time steps. The
reshape can therefore

be written as:

```
...
X = X.reshape((X.shape[0], X.shape[1], 1))

```

We can make this concept concrete with a worked example. The complete code listing is

provided below.

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
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(series.shape)
X, y = split_sequence(series, 3)
print(X.shape, y.shape)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape)
```
Running the example first prints the shape of the univariate time series, in this case 10
time steps. It then summarizes the shape if the input (X) and output (y) elements of each
sample after the univariate series has been converted into a supervised learning problem, in
this case, the data has 7 samples and the input data has 3 features per sample, which we
know are actually time steps. Finally, the input element of each sample is reshaped to be
three-dimensional suitable for fitting an LSTM or CNN and now has the shape[7, 3, 1]or 7
samples, 3 time steps, 1 feature.
```
(10,)
(7, 3) (7,)
(7, 3, 1)

```

### Data Preparation Example

Consider that you are in the current situation:

I have two columns in my data file with 5,000 rows, column 1 is time (with 1 hour
interval) and column 2 is the number of sales and I am trying to forecast the number
of sales for future time steps. Help me to set the number of samples, time steps and
features in this data for an LSTM?

There are few problems here:

- Data Shape. LSTMs expect 3D input, and it can be challenging to get your head around
this the first time.

- Sequence Length. LSTMs don’t like sequences of more than 200-400 time steps, so the
data will need to be split into subsamples.

We will work through this example, broken down into the following 4
steps:

1.  Load the Data
2.  Drop the Time Column
3.  Split Into Samples
4.  Reshape Subsequences

#### Load the Data

We can load this dataset as a PandasSeriesusing the function readcsv().

```

# load time series dataset
series = read_csv('filename.csv', header=0, index_col=0)

```

For this example, we will mock loading by defining a new dataset in memory with 5,000

time steps.

```
from numpy import array
data = list()
n = 5000
for i in range(n):
data.append([i+1, (i+1)*10])
data = array(data)
print(data[:5, :])
print(data.shape)

```

Running this piece both prints the first 5 rows of data and the shape of
the loaded data. We

can see we have 5,000 rows and 2 columns: a standard univariate time
series dataset.

```

[[ 1 10]
[ 2 20]
[ 3 30]
[ 4 40]
[ 5 50]]
(5000, 2)

```

If your time series data is uniform over time and there is no missing
values, we can drop the

time column. If not, you may want to look at imputing the missing
values, resampling the data

to a new time scale, or developing a model that can handle missing
values. Here, we just drop

the first column:

```
from numpy import array
data = list()
n = 5000
for i in range(n):
data.append([i+1, (i+1)*10])
data = array(data)
data = data[:, 1]
print(data.shape)

```

Running the example prints the shape of the dataset after the time
column has been removed.

```

(5000,)

```

#### Split Into Samples

LSTMs need to process samples where each sample is a single sequence of observations. In this
case, 5,000 time steps is too long; LSTMs work better with 200-to-400 time steps. Therefore, we
need to split the 5,000 time steps into multiple shorter sub-sequences. There are many ways to
do this, and you may want to explore some depending on your problem. For example, perhaps

you need overlapping sequences, perhaps non-overlapping is good but your
model needs state

across the sub-sequences and so on. In this example, we will split the 5,000 time steps into 25
sub-sequences of 200 time steps each. Rather than using NumPy or Python tricks, we will do
this the old fashioned way so you can see what is going on.

```

# example of splitting a univariate sequence into subsequences
from numpy import array

# define the dataset
data = list()
n = 5000
for i in range(n):
data.append([i+1, (i+1)*10])
data = array(data)
# drop time
data = data[:, 1]
# split into samples (e.g. 5000/200 = 25)
samples = list()
length = 200
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
# grab from i to i + 200
sample = data[i:i+length]
samples.append(sample)
print(len(samples))

```

We now have 25 subsequences of 200 time steps each.

```
25

```

The LSTM needs data with the format of[samples, timesteps, features]. We
have 25
samples, 200 time steps per sample, and 1 feature. First, we need to convert our list of arrays
into a 2D NumPy array with the shape[25, 200].

```

# example of creating an array of subsequence
from numpy import array

# define the dataset
data = list()
n = 5000
for i in range(n):
data.append([i+1, (i+1)*10])
data = array(data)

```
data = data[:, 1]
samples = list()
length = 200
for i in range(0,n,length):
sample = data[i:i+length]
samples.append(sample)
data = array(samples)
print(data.shape)

```

Running this piece, you should see that we have 25 rows and 200 columns.
Interpreted in a
machine learning context, this dataset has 25 samples and 200 features
per sample.

```
(25, 200)

```

Next, we can use thereshape() function to add one additional dimension
for our single
feature and use the existing columns as time steps instead.

```

from numpy import array
data = list()
n = 5000
for i in range(n):
data.append([i+1, (i+1)*10])
data = array(data)
data = data[:, 1]
samples = list()
length = 200
for i in range(0,n,length):
sample = data[i:i+length]
samples.append(sample)
data = array(samples)
data = data.reshape((len(samples), length, 1))
print(data.shape)
```

And that is it. The data can now be used as an input (X) to an LSTM
model, or even a CNN model.

```
(25, 200, 1)
```

### Extensions

This section lists some ideas for extending the tutorial that you may
wish to explore.

- Explain Data Shape. Explain in your own words the meaning of samples, time steps
and features.

- Worked Example. Select a standard time series forecasting problem and manually
reshape it into a structure suitable for training a CNN or LSTM model.

- Develop Framework. Develop a function to automatically reshape a time series dataset
into samples and into a shape suitable for training a CNN or LSTM model.


### Further Reading

This section provides more resources on the topic if you are looking to
go deeper.

- numpy.reshapeAPI.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

- Keras Recurrent Layers API in Keras.
https://keras.io/layers/recurrent/

- Keras Convolutional Layers API in Keras.
https://keras.io/layers/convolutional/

### Summary

In this tutorial, you discovered exactly how to transform a time series data set into a three-
dimensional structure ready for fitting a CNN or LSTM model.
Specifically, you learned:

- How to transform a time series dataset into a two-dimensional
supervised learning format.

- How to transform a two-dimensional time series dataset into a three-dimensional structure
suitable for CNNs and LSTMs.

- How to step through a worked example of splitting a very long time series into subsequences
ready for training a CNN or LSTM model.

#### Next

In the next lab, you will discover how to develop Multilayer Perceptron models for time series
forecasting.
