<img align="right" src="../logo-small.png">


# How to Develop LSTMs for Human Activity Recognition

Human activity recognition is the problem of classifying sequences of accelerometer data
recorded by specialized harnesses or smartphones into known well-defined movements. Classical
approaches to the problem involve hand crafting features from the time series data based on
fixed-sized windows and training machine learning models, such as ensembles of decision trees.
The difficulty is that this feature engineering requires strong expertise in the field. Recently,
deep learning methods such as recurrent neural networks like as LSTMs and variations that
make use of one-dimensional convolutional neural networks or CNNs have been shown to provide
state-of-the-art results on challenging activity recognition tasks with little or no data feature
engineering, instead using feature learning on raw data. In this tutorial, you will discover
three recurrent neural network architectures for modeling an activity recognition time series
classification problem. After completing this tutorial, you will know:

- How to develop a Long Short-Term Memory Recurrent Neural Network for human activity
recognition.
- How to develop a one-dimensional Convolutional Neural Network LSTM, or CNN-LSTM,
model.
- How to develop a one-dimensional Convolutional LSTM, or ConvLSTM, model for the
same problem.
Let’s get started.

#### Tutorial Overview
This tutorial is divided into four parts; they are:
1. Activity Recognition Using Smartphones Dataset
2. LSTM Model
3. CNN-LSTM Model
4. ConvLSTM Model


#### Activity Recognition Using Smartphones Dataset
Human Activity Recognition, or HAR for short, is the problem of predicting what a person is
doing based on a trace of their movement using sensors. A standard human activity recognition
dataset is the Activity Recognition Using Smartphones made available in 2012. For more
information on this dataset, see Chapter 22. The data is provided as a single zip file that is
about 58 megabytes in size. A direct for downloading the dataset is provided below:
- HAR Smartphones.zip 1
Download the dataset and unzip all files into a new directory in your current working
directory named HARDataset.

#### LSTM Model
In this section, we will develop a Long Short-Term Memory network model (LSTM) for the
human activity recognition dataset. LSTM network models are a type of recurrent neural
network that are able to learn and remember over long sequences of input data. They are
intended for use with data that is comprised of long sequences of data, up to 200 to 400 time
steps. They may be a good fit for this problem. The model can support multiple parallel
sequences of input data, such as each axis of the accelerometer and gyroscope data. The model
learns to extract features from sequences of observations and how to map the internal features
to different activity types.

The benefit of using LSTMs for sequence classification is that they can learn from the raw
time series data directly, and in turn do not require domain expertise to manually engineer
input features. The model can learn an internal representation of the time series data and
ideally achieve comparable performance to models fit on a version of the dataset with engineered
features. For more information on LSTMs for time series forecasting, see Chapter 9. This
section is divided into four parts; they are:
1. Load Data
2. Fit and Evaluate Model
3. Summarize Results
4. Complete Example

#### Load Data
The first step is to load the raw dataset into memory. There are three main signal types in the
raw data: total acceleration, body acceleration, and body gyroscope. Each has three axes of
data. This means that there are a total of nine variables for each time step. Further, each series
of data has been partitioned into overlapping windows of 2.65 seconds of data, or 128 time steps.
These windows of data correspond to the windows of engineered features (rows) in the previous
section.

This means that one row of data has (128 × 9), or 1,152, elements. This is a little less than
double the size of the 561 element vectors in the previous section and it is likely that there
is some redundant data. The signals are stored in the /Inertial Signals/ directory under
the train and test subdirectories. Each axis of each signal is stored in a separate file, meaning
that each of the train and test datasets have nine input files to load and one output file to load.
We can batch the loading of these files into groups given the consistent directory structures
and file naming conventions. The input data is in CSV format where columns are separated by
whitespace. Each of these files can be loaded as a NumPy array. The load file() function
below loads a dataset given the file path to the file and returns the loaded data as a NumPy
array.

```
# load a single file as a numpy array
def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values
```

We can then load all data for a given group (train or test) into a single three-dimensional
NumPy array, where the dimensions of the array are [samples, timesteps, features]. To
make this clearer, there are 128 time steps and nine features, where the number of samples is the
number of rows in any given raw signal data file. The load group() function below implements
this behavior. The dstack() NumPy function allows us to stack each of the loaded 3D arrays
into a single 3D array where the variables are separated on the third dimension (features).

```
# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
loaded = list()
for name in filenames:
data = load_file(prefix + name)
loaded.append(data)
# stack group so that features are the 3rd dimension
loaded = dstack(loaded)
return loaded
```

We can use this function to load all input signal data for a given group, such as train or test.
The load dataset group() function below loads all input signal data and the output data for
a single group using the consistent naming conventions between the train and test directories.

```
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
filepath = prefix + group + '/Inertial Signals/'
# load all 9 files as a single array
filenames = list()
# total acceleration
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt',
'total_acc_z_'+group+'.txt']
# body acceleration
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt',
'body_acc_z_'+group+'.txt']
# body gyroscope
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt',
'body_gyro_z_'+group+'.txt']
# load input data
X = load_group(filenames, filepath)
# load class output
y = load_file(prefix + group + '/y_'+group+'.txt')
return X, y
```

Finally, we can load each of the train and test datasets. The output data is defined as an
integer for the class number. We must one hot encode these class integers so that the data is
suitable for fitting a neural network multiclass classification model. We can do this by calling
the to categorical() Keras function. The load dataset() function below implements this
behavior and returns the train and test X and y elements ready for fitting and evaluating the
defined models.

```
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
print(trainX.shape, trainy.shape)
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
print(testX.shape, testy.shape)
# zero-offset class values
trainy = trainy - 1
testy = testy - 1
# one hot encode y
trainy = to_categorical(trainy)
testy = to_categorical(testy)
print(trainX.shape, trainy.shape, testX.shape, testy.shape)
return trainX, trainy, testX, testy
```

#### Fit and Evaluate Model
Now that we have the data loaded into memory ready for modeling, we can define, fit, and
evaluate an LSTM model. We can define a function named evaluate model() that takes the
train and test dataset, fits a model on the training dataset, evaluates it on the test dataset, and
returns an estimate of the model’s performance. First, we must define the LSTM model using
the Keras deep learning library. The model requires a three-dimensional input with [samples,
timesteps, features].

This is exactly how we have loaded the data, where one sample is one window of the time
series data, each window has 128 time steps, and a time step has nine variables or features. The
output for the model will be a six-element vector containing the probability of a given window
belonging to each of the six activity types. The input and output dimensions are required when
fitting the model, and we can extract them from the provided training dataset.

```
# define data shape
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
```

he model is defined as a Sequential Keras model, for simplicity.
We will define the model as having a single LSTM hidden layer. This is followed by a dropout
layer intended to reduce overfitting of the model to the training data. Finally, a dense fully
connected layer is used to interpret the features extracted by the LSTM hidden layer, before a
final output layer is used to make predictions. The efficient Adam version of stochastic gradient
descent will be used to optimize the network, and the categorical cross entropy loss function
will be used given that we are learning a multiclass classification problem. The definition of the
model is listed below.

```
# define the lstm model
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

The model is fit for a fixed number of epochs, in this case 15, and a batch size of 64 samples
will be used, where 64 windows of data will be exposed to the model before the weights of
the model are updated. Once the model is fit, it is evaluated on the test dataset and the
accuracy of the fit model on the test dataset is returned. Note, it is common to not shuffle
sequence data when fitting an LSTM. Here we do shuffle the windows of input data during
training (the default). In this problem, we are interested in harnessing the LSTMs ability to
learn and extract features across the time steps in a window, not across windows. The complete
evaluate model() function is listed below.

```
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
verbose, epochs, batch_size = 0, 15, 64
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
return accuracy
```

There is nothing special about the network structure or chosen hyperparameters, they are
just a starting point for this problem.

#### Summarize Results
We cannot judge the skill of the model from a single evaluation. The reason for this is that
neural networks are stochastic, meaning that a different specific model will result when training
the same model configuration on the same data. This is a feature of the network in that it gives
the model its adaptive ability, but requires a slightly more complicated evaluation of the model.
We will repeat the evaluation of the model multiple times, then summarize the performance of
the model across each of those runs. For example, we can call evaluate model() a total of 10
times. This will result in a population of model evaluation scores that must be summarized.

```
# repeat experiment
scores = list()
for r in range(repeats):
score = evaluate_model(trainX, trainy, testX, testy)
score = score * 100.0
print('>#%d: %.3f' % (r+1, score))
scores.append(score)
```

We can summarize the sample of scores by calculating and reporting the mean and standard
deviation of the performance. The mean gives the average accuracy of the model on the dataset,
whereas the standard deviation gives the average variance of the accuracy from the mean. The
function summarize results() below summarizes the results of a run.

```
# summarize scores
def summarize_results(scores):
print(scores)
m, s = mean(scores), std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
```

We can bundle up the repeated evaluation, gathering of results, and summarization of results
into a main function for the experiment, called run experiment(), listed below. By default,
the model is evaluated 10 times before the performance of the model is reported.

```
# run an experiment
def run_experiment(repeats=10):
# load data
trainX, trainy, testX, testy = load_dataset()
# repeat experiment
scores = list()
for r in range(repeats):
score = evaluate_model(trainX, trainy, testX, testy)
score = score * 100.0
print('>#%d: %.3f' % (r+1, score))
scores.append(score)
# summarize results
summarize_results(scores)
```


#### Complete Example
Now that we have all of the pieces, we can tie them together into a worked example. The
complete code listing is provided below.

```
# lstm model for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
# load a single file as a numpy array
def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
loaded = list()
for name in filenames:
data = load_file(prefix + name)
loaded.append(data)
# stack group so that features are the 3rd dimension
loaded = dstack(loaded)
return loaded
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
filepath = prefix + group + '/Inertial Signals/'
# load all 9 files as a single array
filenames = list()
# total acceleration
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt',
'total_acc_z_'+group+'.txt']
# body acceleration
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt',
'body_acc_z_'+group+'.txt']
# body gyroscope
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt',
'body_gyro_z_'+group+'.txt']
# load input data
X = load_group(filenames, filepath)
# load class output
y = load_file(prefix + group + '/y_'+group+'.txt')
return X, y
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
# zero-offset class values
trainy = trainy - 1
testy = testy - 1
# one hot encode y
trainy = to_categorical(trainy)
testy = to_categorical(testy)
return trainX, trainy, testX, testy
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
verbose, epochs, batch_size = 0, 15, 64
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
return accuracy
# summarize scores
def summarize_results(scores):
print(scores)
m, s = mean(scores), std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# load data
trainX, trainy, testX, testy = load_dataset()
# repeat experiment
scores = list()
for r in range(repeats):
score = evaluate_model(trainX, trainy, testX, testy)
score = score * 100.0
print('>#%d: %.3f' % (r+1, score))
scores.append(score)
# summarize results
summarize_results(scores)
# run the experiment
run_experiment()
```

Running the example first loads the dataset. The models are created and evaluated and a
debug message is printed for each. Finally, the sample of scores is printed, followed by the mean
and standard deviation. We can see that the model performed well, achieving a classification
accuracy of about 89.7% trained on the raw dataset, with a standard deviation of about 1.3.
This is a good result, considering that the original paper published a result of 89%, trained onthe dataset with heavy domain-specific feature engineering, not the raw dataset.

**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
>#1: 90.058
>#2: 85.918
>#3: 90.974
>#4: 89.515
>#5: 90.159
>#6: 91.110
>#7: 89.718
>#8: 90.295
>#9: 89.447
>#10: 90.024
[90.05768578215134, 85.91788259246692, 90.97387173396675, 89.51476077366813,
90.15948422124194, 91.10960298608755, 89.71835765184933, 90.29521547336275,
89.44689514760775, 90.02375296912113]
Accuracy: 89.722% (+/-1.371)
```

Now that we have seen how to develop an LSTM model for time series classification, let’s
look at how we can develop a more sophisticated CNN-LSTM model.

####  CNN-LSTM Model
The CNN-LSTM architecture involves using Convolutional Neural Network (CNN) layers for
feature extraction on input data combined with LSTMs to support sequence prediction. For
more information on the use of CNN-LSTM models for time series forecasting, see Chapter 9.
The CNN-LSTM model will read subsequences of the main sequence in as blocks, extract
features from each block, then allow the LSTM to interpret the features extracted from each
block. One approach to implementing this model is to split each window of 128 time steps into
subsequences for the CNN model to process. For example, the 128 time steps in each window
can be split into four subsequences of 32 time steps.

```
# reshape data into time steps of sub-sequences
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
```

We can then define a CNN model that expects to read in sequences with a length of 32 time
steps and nine features. The entire CNN model can be wrapped in a TimeDistributed layer
to allow the same CNN model to read in each of the four subsequences in the window. The
extracted features are then flattened and provided to the LSTM model to read, extracting its
own features before a final mapping to an activity is made.


```
# define cnn-lstm model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
```

It is common to use two consecutive CNN layers followed by dropout and a max pooling
layer, and that is the simple structure used in the this CNN-LSTM model. The updated
evaluate model() is listed below.

```
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
# define model
verbose, epochs, batch_size = 0, 25, 64
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# reshape data into time steps of sub-sequences
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
return accuracy
```

We can evaluate this model as we did the straight LSTM model in the previous section. The
complete code listing is provided below.

```
# cnn lstm model for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
# load a single file as a numpy array
def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
loaded = list()
for name in filenames:
data = load_file(prefix + name)
loaded.append(data)
# stack group so that features are the 3rd dimension
loaded = dstack(loaded)
return loaded
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
filepath = prefix + group + '/Inertial Signals/'
# load all 9 files as a single array
filenames = list()
# total acceleration
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt',
'total_acc_z_'+group+'.txt']
# body acceleration
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt',
'body_acc_z_'+group+'.txt']
# body gyroscope
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt',
'body_gyro_z_'+group+'.txt']
# load input data
X = load_group(filenames, filepath)
# load class output
y = load_file(prefix + group + '/y_'+group+'.txt')
return X, y
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
# zero-offset class values
trainy = trainy - 1
testy = testy - 1
# one hot encode y
trainy = to_categorical(trainy)
testy = to_categorical(testy)
return trainX, trainy, testX, testy
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
# define model
verbose, epochs, batch_size = 0, 25, 64
n_features, n_outputs = trainX.shape[2], trainy.shape[1]
# reshape data into time steps of sub-sequences
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
return accuracy
# summarize scores
def summarize_results(scores):
print(scores)
m, s = mean(scores), std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# load data
trainX, trainy, testX, testy = load_dataset()
# repeat experiment
scores = list()
for r in range(repeats):
score = evaluate_model(trainX, trainy, testX, testy)
score = score * 100.0
print('>#%d: %.3f' % (r+1, score))
scores.append(score)
# summarize results
summarize_results(scores)
# run the experiment
run_experiment()
```

Running the example summarizes the model performance for each of the 10 runs before
a final summary of the model’s performance on the test set is reported. We can see that the
model achieved a performance of about 90.6% with a standard deviation of about 1%.
**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
>#1: 91.517
>#2: 91.042
>#3: 90.804
>#4: 92.263
>#5: 89.684
>#6: 88.666
>#7: 91.381
>#8: 90.804
>#9: 89.379
>#10: 91.347
[91.51679674244994, 91.04173736002714, 90.80420766881574, 92.26331862911435,
89.68442483881914, 88.66644044791313, 91.38106549032915, 90.80420766881574,
89.37902952154734, 91.34713267729894]
Accuracy: 90.689% (+/-1.051)
```


####  ConvLSTM Model
A further extension of the CNN-LSTM idea is to perform the convolutions of the CNN (e.g.
how the CNN reads the input sequence data) as part of the LSTM. This combination is called
a Convolutional LSTM, or ConvLSTM for short, and like the CNN-LSTM is also used for
spatiotemporal data. The ConvLSTM2D class, by default, expects input data to have the shape:
[samples, time, rows, cols, channels]. Where each time step of data is defined as an
image of (rows × columns) data points.

In the previous section, we divided a given window of data (128 time steps) into four
subsequences of 32 time steps. We can use this same subsequence approach in defining the
ConvLSTM2D input where the number of time steps is the number of subsequences in the window,
the number of rows is 1 as we are working with one-dimensional data, and the number of
columns represents the number of time steps in the subsequence, in this case 32. For this chosen
framing of the problem, the input for the ConvLSTM2D would therefore be:

- Samples: n, for the number of windows in the dataset.
- Time: 4, for the four subsequences that we split a window of 128 time steps into.
- Rows: 1, for the one-dimensional shape of each subsequence.
- Columns: 32, for the 32 time steps in an input subsequence.
- Channels: 9, for the nine input variables.

We can now prepare the data for the ConvLSTM2D model.

```
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# reshape into subsequences (samples, timesteps, rows, cols, channels)
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
```

The ConvLSTM2D class requires configuration both in terms of the CNN and the LSTM. This
includes specifying the number of filters (e.g. 64), the two-dimensional kernel size, in this case
(1 row and 3 columns of the subsequence time steps), and the activation function, in this case
rectified linear. As with a CNN or LSTM model, the output must be flattened into one long
vector before it can be interpreted by a dense layer.

```
# define convlstm model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu',
input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
```

We can then evaluate the model as we did the LSTM and CNN-LSTM models before it.
The complete example is listed below.

```
# convlstm model for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
# load a single file as a numpy array
def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
loaded = list()
for name in filenames:
data = load_file(prefix + name)
loaded.append(data)
# stack group so that features are the 3rd dimension
loaded = dstack(loaded)
return loaded
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
filepath = prefix + group + '/Inertial Signals/'
# load all 9 files as a single array
filenames = list()
# total acceleration
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt',
'total_acc_z_'+group+'.txt']
# body acceleration
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt',
'body_acc_z_'+group+'.txt']
# body gyroscope
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt',
'body_gyro_z_'+group+'.txt']
# load input data
X = load_group(filenames, filepath)
# load class output
y = load_file(prefix + group + '/y_'+group+'.txt')
return X, y
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
# zero-offset class values
trainy = trainy - 1
testy = testy - 1
# one hot encode y
trainy = to_categorical(trainy)
testy = to_categorical(testy)
return trainX, trainy, testX, testy
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
# define model
verbose, epochs, batch_size = 0, 25, 64
n_features, n_outputs = trainX.shape[2], trainy.shape[1]
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu',
input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
return accuracy
# summarize scores
def summarize_results(scores):
print(scores)
m, s = mean(scores), std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# load data
trainX, trainy, testX, testy = load_dataset()
# repeat experiment
scores = list()
for r in range(repeats):
score = evaluate_model(trainX, trainy, testX, testy)
score = score * 100.0
print('>#%d: %.3f' % (r+1, score))
scores.append(score)
# summarize results
summarize_results(scores)
# run the experiment
run_experiment()
```

As with the prior experiments, running the model prints the performance of the model each
time it is fit and evaluated. A summary of the final model performance is presented at the end
of the run. We can see that the model does consistently perform well on the problem achieving
an accuracy of about 90%, perhaps with fewer resources than the larger CNN-LSTM model.
**Note:** Given the stochastic nature of the algorithm, your specific results may vary. Consider
running the example a few times.

```
>#1: 90.092
>#2: 91.619
>#3: 92.128
>#4: 90.533
>#5: 89.243
>#6: 90.940
>#7: 92.026
>#8: 91.008
>#9: 90.499
>#10: 89.922
[90.09161859518154, 91.61859518154056, 92.12758737699356, 90.53274516457415,
89.24329826942655, 90.93993892093654, 92.02578893790296, 91.00780454699695,
90.49881235154395, 89.92195453003053]
Accuracy: 90.801% (+/-0.886)
```


####  Extensions
This section lists some ideas for extending the tutorial that you may wish to explore.
- Data Preparation. Consider exploring whether simple data scaling schemes can further
lift model performance, such as normalization, standardization, and power transforms.
- LSTM Variations. There are variations of the LSTM architecture that may achieve
better performance on this problem, such as stacked LSTMs and Bidirectional LSTMs.
- Hyperparameter Tuning. Consider exploring tuning of model hyperparameters such
as the number of units, training epochs, batch size, and more.


####  Further Reading
This section provides more resources on the topic if you are looking to go deeper.
- Getting started with the Keras Sequential model.
https://keras.io/getting-started/sequential-model-guide/
- Getting started with the Keras functional API.
https://keras.io/getting-started/functional-api-guide/
- Keras Sequential Model API.
https://keras.io/models/sequential/
- Keras Core Layers API.
https://keras.io/layers/core/
- Keras Convolutional Layers API.
https://keras.io/layers/convolutional/
- Keras Pooling Layers API.
https://keras.io/layers/pooling/
- Keras Recurrent Layers API.
https://keras.io/layers/recurrent/

#### Summary
In this tutorial, you discovered three recurrent neural network architectures for modeling an
activity recognition time series classification problem. Specifically, you learned:
- How to develop a Long Short-Term Memory Recurrent Neural Network for human activity
recognition.
- How to develop a one-dimensional Convolutional Neural Network LSTM, or CNN-LSTM,
model.
- How to develop a one-dimensional Convolutional LSTM, or ConvLSTM, model for the
same problem.