
<img align="right" src="../logo-small.png">


### How to Develop ML Models for Human Activity Recognition

Human activity recognition is the problem of classifying sequences of accelerometer data
recorded by specialized harnesses or smartphones into known well-defined movements. Classical
approaches to the problem involve hand crafting features from the time series data based on
fixed-sized windows and training machine learning models, such as ensembles of decision trees.
The difficulty is that this feature engineering requires deep expertise in the field.
Recently, deep learning methods such as recurrent neural networks and one-dimensional
convolutional neural networks, or CNNs, have been shown to provide state-of-the-art results on
challenging activity recognition tasks with little or no data feature engineering, instead using
feature learning on raw data. In this tutorial, you will discover how to evaluate a diverse suite
of machine learning algorithms on the Activity Recognition Using Smartphones dataset. After
completing this tutorial, you will know:

- How to load and evaluate nonlinear and ensemble machine learning algorithms on the
feature-engineered version of the activity recognition dataset.
- How to load and evaluate machine learning algorithms on the raw signal data for the
activity recognition dataset.
- How to define reasonable lower and upper bounds on the expected performance of more
sophisticated algorithms capable of feature learning, such as deep learning methods.

Let’s get started.

#### Tutorial Overview
This tutorial is divided into three parts; they are:
1. Activity Recognition Using Smartphones Dataset
2. Modeling Feature Engineered Data
3. Modeling Raw Data

##### Activity Recognition Using Smartphones Dataset
Human Activity Recognition, or HAR for short, is the problem of predicting what a person is
doing based on a trace of their movement using sensors. A standard human activity recognition
dataset is the Activity Recognition Using Smartphones made available in 2012. For more
information on this dataset, see Chapter 22. The data is provided as a single zip file that is
about 58 megabytes in size. A direct for downloading the dataset is provided below:
- HAR Smartphones.zip 1
Download the dataset and unzip all files into a new directory in your current working
directory named HARDataset.

##### Modeling Feature Engineered Data
In this section, we will develop code to load the feature-engineered version of the dataset and
evaluate a suite of nonlinear machine learning algorithms, including SVM used in the original
paper. The goal is to achieve at least 89% accuracy on the test dataset. The results of methods
using the feature-engineered version of the dataset provide a baseline for any methods developed
for the raw data version. This section is divided into five parts; they are:

1. Load Dataset
2. Define Models
3. Evaluate Models
4. Summarize Results
5. Complete Example

#### Load Dataset
The first step is to load the train and test input (X) and output (y) data. Specifically, the
following files:
- HARDataset/train/X train.txt
- HARDataset/train/y train.txt
- HARDataset/test/X test.txt
- HARDataset/test/y test.txt

The input data is in CSV format where columns are separated via whitespace. Each of these
files can be loaded as a NumPy array. The load file() function bel

```
# load a single file as a numpy array
def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values

```

We can call this function to load the X and y files for a given train or test set group, given
the similarity in directory layout and filenames. The load dataset group() function below
will load both of these files for a group and return the X and y elements as NumPy arrays.
This function can then be used to load the X and y elements for both the train and test groups.

```
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
# load input data
X = load_file(prefix + group +'/X_'+group+'.txt')
# load class output
y = load_file(prefix + group +'/y_'+group+'.txt')
return X, y

```

Finally, we can load both the train and test dataset and return them as NumPy arrays ready
for fitting and evaluating machine learning models.

```
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
print(trainX.shape, trainy.shape)
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
print(testX.shape, testy.shape)
# flatten y
trainy, testy = trainy[:,0], testy[:,0]
print(trainX.shape, trainy.shape, testX.shape, testy.shape)
return trainX, trainy, testX, testy

```
We can call this function to load all of the required data; for example:

```
# load dataset
trainX, trainy, testX, testy = load_dataset()

```
#### Define Models

Next, we can define a list of machine learning models to evaluate on this problem. We will
evaluate the models using default configurations. We are not looking for optimal configurations
of these models at this point, just a general idea of how well sophisticated models with default
configurations perform on this problem. We will evaluate a diverse set of nonlinear and ensemble
machine learning algorithms, specifically:

**Nonlinear Algorithms:**

- k-Nearest Neighbors

- Classification and Regression Tree

- Support Vector Machine

- Naive Bayes

**Ensemble Algorithms:**

- Bagged Decision Trees

- Random Forest

- Extra Trees

- Gradient Boosting Machine

We will define the models and store them in a dictionary that maps the model object to a
short name that will help in analyzing the results. Thedefinemodels() function below defines
the eight models that we will evaluate.

```
# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
# nonlinear models
models['knn'] = KNeighborsClassifier(n_neighbors=7)
models['cart'] = DecisionTreeClassifier()
models['svm'] = SVC()
models['bayes'] = GaussianNB()
# ensemble models
models['bag'] = BaggingClassifier(n_estimators=100)
models['rf'] = RandomForestClassifier(n_estimators=100)
models['et'] = ExtraTreesClassifier(n_estimators=100)
models['gbm'] = GradientBoostingClassifier(n_estimators=100)
print('Defined %d models'% len(models))
return models

```

This function is quite extensible and you can easily update to define any machine learning
models or model configurations you wish.

#### Evaluate Models

The next step is to evaluate the defined models in the loaded dataset.
This step is divided into
the evaluation of a single model and the evaluation of all of the models. We will evaluate a
single model by first fitting it on the training dataset, making a prediction on the test dataset,
and then evaluating the prediction using a metric. In this case we will use classification accuracy
that will capture the performance (or error) of a model given the balance observations across
the six activities (or classes). Theevaluatemodel() function below implements this behavior,
evaluating a given model and returning the classification accuracy as a percentage.


```
# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
# fit the model
model.fit(trainX, trainy)
# make predictions
yhat = model.predict(testX)
# evaluate predictions
accuracy = accuracy_score(testy, yhat)
return accuracy * 100.0

```

We can now call the evaluate model() function repeatedly for each of the defined model.
The evaluate models() function below implements this behavior, taking the dictionary of
defined models, and returns a dictionary of model names mapped to their classification accuracy.
Because the evaluation of the models may take a few minutes, the function prints the performance
of each model after it is evaluated as some verbose feedback.

```
# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
results = dict()
for name, model in models.items():
# evaluate the model
results[name] = evaluate_model(trainX, trainy, testX, testy, model)
# show process
print('>%s: %.3f' % (name, results[name]))
return results

```

#### Summarize Results

The final step is to summarize the findings. We can sort all of the results by the classification
accuracy in descending order because we are interested in maximizing accuracy. The results
of the evaluated models can then be printed, clearly showing the relative rank of each of the
evaluated models. The summarize results() function below implements this behavior.

```
# print and plot the results
def summarize_results(results, maximize=True):
# create a list of (name, mean(scores)) tuples
mean_scores = [(k,v) for k,v in results.items()]
# sort tuples by mean score
mean_scores = sorted(mean_scores, key=lambda x: x[1])
# reverse for descending order (e.g. for accuracy)
if maximize:
mean_scores = list(reversed(mean_scores))
print()
for name, score in mean_scores:
print('Name=%s, Score=%.3f'% (name, score))

```

#### Complete Example

We know that we have all of the pieces in place. The complete example of
evaluating a suite of
eight machine learning models on the feature-engineered version of the dataset is listed below.

```
# spot check ml algorithms on engineered-features from the har dataset
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load a single file as a numpy array
def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
# load input data
X = load_file(prefix + group +'/X_'+group+'.txt')
# load class output
y = load_file(prefix + group +'/y_'+group+'.txt')
return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
# flatten y
trainy, testy = trainy[:,0], testy[:,0]
return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
# nonlinear models
models['knn'] = KNeighborsClassifier(n_neighbors=7)
models['cart'] = DecisionTreeClassifier()
models['svm'] = SVC()
models['bayes'] = GaussianNB()
# ensemble models
models['bag'] = BaggingClassifier(n_estimators=100)
models['rf'] = RandomForestClassifier(n_estimators=100)
models['et'] = ExtraTreesClassifier(n_estimators=100)
models['gbm'] = GradientBoostingClassifier(n_estimators=100)
print('Defined %d models'% len(models))
return models

# evaluate a single model


def evaluate_model(trainX, trainy, testX, testy, model):
model.fit(trainX, trainy)

yhat = model.predict(testX)

accuracy = accuracy_score(testy, yhat)
return accuracy * 100.0

def evaluate_models(trainX, trainy, testX, testy, models):
results = dict()
for name, model in models.items():

results[name] = evaluate_model(trainX, trainy, testX, testy, model)
print('>%s: %.3f' % (name, results[name]))
return results

def summarize_results(results, maximize=True):

mean_scores = [(k,v) for k,v in results.items()]

mean_scores = sorted(mean_scores, key=lambda x: x[1])

if maximize:
mean_scores = list(reversed(mean_scores))
print()
for name, score in mean_scores:
print('Name=%s, Score=%.3f'% (name, score))
trainX, trainy, testX, testy = load_dataset()
models = define_models()

results = evaluate_models(trainX, trainy, testX, testy, models)

summarize_results(results)

```

Running the example first loads the train and test datasets. The eight models are then
evaluated in turn, printing the performance for each. Finally, a rank of the models by their
performance on the test set is displayed. We can see that both the ExtraTrees ensemble method
and the Support Vector Machines nonlinear methods achieve a performance of about 94%
accuracy on the test set. This is a great result, exceeding the reported 89% by SVM in the
original paper.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.

```

Defined 8 models


>knn: 90.329
>cart: 86.020
>svm: 94.028
>bayes: 77.027
>bag: 89.820
>rf: 92.772
>et: 94.028
>gbm: 93.756

Name=et, Score=94.028
Name=svm, Score=94.028
Name=gbm, Score=93.756
Name=rf, Score=92.772
Name=knn, Score=90.329
Name=bag, Score=89.820
Name=cart, Score=86.020
Name=bayes, Score=77.027

```

These results show what is possible given domain expertise in the preparation of the data and
the engineering of domain-specific features. As such, these results can be taken as a performance
upper-bound of what could be pursued through more advanced methods that may be able to
automatically learn features as part of fitting the model, such as deep learning methods. Any
such advanced methods would be fit and evaluated on the raw data from which the engineered
features were derived. And as such, the performance of machine learning algorithms evaluated
on that data directly may provide an expected lower bound on the performance of any more
advanced methods. We will explore this in the next section.

#### Modeling Raw Data

We can use the same framework for evaluating machine learning models on the raw data. The
raw data does require some more work to load. There are three main signal types in the raw
data: total acceleration, body acceleration, and body gyroscope. Each has three axes of data.
This means that there are a total of nine variables for each time step. Further, each series of
data has been partitioned into overlapping windows of 2.65 seconds of data, or 128 time steps.
These windows of data correspond to the windows of engineered features (rows) in the previous
section.

This means that one row of data has 128 × 9 or 1,152 elements. This is a little less than
double the size of the 561 element vectors in the previous section and it is likely that there is
some redundant data. The signals are stored in the /Inertial Signals/ directory under the
train and test subdirectories. Each axis of each signal is stored in a separate file, meaning that
each of the train and test datasets have nine input files to load and one output file to load. We
can batch the loading of these files into groups given the consistent directory structures and file
naming conventions.

First, we can load all data for a given group into a single three-dimensional NumPy array,
where the dimensions of the array are [samples, timesteps, features]. To make this clearer,
there are 128 time steps and nine features, where the number of samples is the number of rows
in any given raw signal data file. The load group() function below implements this behavior.
The dstack() NumPy function allows us to stack each of the loaded 3D arrays into a single 3D
array where the variables are separated on the third dimension (features).


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
a single group using the consistent naming conventions between the directories.

```
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
filepath = prefix + group + '/Inertial Signals/'
# load all 9 files as a single array
filenames = list()
# total acceleration
filenames += ['total_acc_x_'+group+'.txt','total_acc_y_'+group+'.txt',
'total_acc_z_'+group+'.txt']
# body acceleration
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt',
'body_acc_z_'+group+'.txt']
# body gyroscope
filenames += ['body_gyro_x_'+group+'.txt','body_gyro_y_'+group+'.txt',
'body_gyro_z_'+group+'.txt']
# load input data
X = load_group(filenames, filepath)
# load class output
y = load_file(prefix + group +'/y_'+group+'.txt')
return X, y

```


Finally, we can load each of the train and test datasets. As part of preparing the loaded data,
we must flatten the windows and features into one long vector. We can do this with the NumPy
reshape function and convert the three dimensions of [samples, timesteps, features] into
the two dimensions of [samples, timesteps × features]. The load dataset() function
below implements this behavior and returns the train and test X and y elements ready for
fitting and evaluating the defined models.

```
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
# load all train
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
print(trainX.shape, trainy.shape)
# load all test
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
print(testX.shape, testy.shape)

# flatten X
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
# flatten y
trainy, testy = trainy[:,0], testy[:,0]
print(trainX.shape, trainy.shape, testX.shape, testy.shape)
return trainX, trainy, testX, testy

```
Putting this all together, the complete example is listed below.

```

from numpy import dstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

def load_file(filepath):
dataframe = read_csv(filepath, header=None, delim_whitespace=True)
return dataframe.values

def load_group(filenames, prefix=''):
loaded = list()
for name in filenames:
data = load_file(prefix + name)
loaded.append(data)

loaded = dstack(loaded)
return loaded

def load_dataset_group(group, prefix=''):
filepath = prefix + group + '/Inertial Signals/'

filenames = list()

filenames +=
['total_acc_x_'+group+'.txt','total_acc_y_'+group+'.txt',
'total_acc_z_'+group+'.txt']

filenames += ['body_acc_x_'+group+'.txt',
'body_acc_y_'+group+'.txt',
'body_acc_z_'+group+'.txt']
filenames +=
['body_gyro_x_'+group+'.txt','body_gyro_y_'+group+'.txt',
'body_gyro_z_'+group+'.txt']

X = load_group(filenames, filepath)


y = load_file(prefix + group +'/y_'+group+'.txt')
return X, y

def load_dataset(prefix=''):
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] *
trainX.shape[2]))
testX = testX.reshape((testX.shape[0], testX.shape[1] *
testX.shape[2]))
trainy, testy = trainy[:,0], testy[:,0]
return trainX, trainy, testX, testy

def define_models(models=dict()):

models['knn'] = KNeighborsClassifier(n_neighbors=7)
models['cart'] = DecisionTreeClassifier()
models['svm'] = SVC()
models['bayes'] = GaussianNB()

models['bag'] = BaggingClassifier(n_estimators=100)
models['rf'] = RandomForestClassifier(n_estimators=100)
models['et'] = ExtraTreesClassifier(n_estimators=100)
models['gbm'] = GradientBoostingClassifier(n_estimators=100)
print('Defined %d models'% len(models))
return models

def evaluate_model(trainX, trainy, testX, testy, model):
model.fit(trainX, trainy)

yhat = model.predict(testX)

accuracy = accuracy_score(testy, yhat)
return accuracy * 100.0

def evaluate_models(trainX, trainy, testX, testy, models):
results = dict()
for name, model in models.items():

results[name] = evaluate_model(trainX, trainy, testX, testy, model)
print('>%s: %.3f' % (name, results[name]))
return results

def summarize_results(results, maximize=True):

mean_scores = [(k,v) for k,v in results.items()]


mean_scores = sorted(mean_scores, key=lambda x: x[1])
# reverse for descending order (e.g. for accuracy)
if maximize:
mean_scores = list(reversed(mean_scores))
print()
for name, score in mean_scores:
print('Name=%s, Score=%.3f'% (name, score))
trainX, trainy, testX, testy = load_dataset()
models = define_models()

results = evaluate_models(trainX, trainy, testX, testy, models)

summarize_results(results)

```

Running the example first loads the dataset. Next the eight defined models are evaluated in
turn. The final results suggest that ensembles of decision trees perform the best on the raw
data. Gradient Boosting and Extra Trees perform the best with about 87% and 86% accuracy,
about seven points below the best performing models on the feature-engineered version of
the dataset. It is encouraging that the Extra Trees ensemble method performed well on both
datasets; it suggests it and similar tree ensemble methods may be suited to the problem, at
least in this simplified framing. We can also see the drop of SVM to about 72% accuracy. The
good performance of ensembles of decision trees may suggest the need for feature selection and
the ensemble methods ability to select those features that are most relevant to predicting the
associated activity.

**Note:** Given the stochastic nature of the algorithm, your specific
results may vary. Consider running the example a few times.

```

Defined 8 models

> knn: 61.893
>  cart: 72.141
>  svm: 76.960
>  bayes: 72.480
>  bag: 84.527
>  rf: 84.662
>  et: 86.902
>  gbm: 87.615

Name=gbm, Score=87.615
Name=et, Score=86.902
Name=rf, Score=84.662
Name=bag, Score=84.527
Name=svm, Score=76.960
Name=bayes, Score=72.480
Name=cart, Score=72.141
Name=knn, Score=61.893


```

As noted in the previous section, these results provide a lower-bound on accuracy for any
more sophisticated methods that may attempt to learn higher order features automatically (e.g.
via feature learning in deep learning methods) from the raw data. In summary, the bounds for
such methods extend on this dataset from about 87% accuracy with GBM on the raw data to
about 94% with Extra Trees and SVM on the highly processed dataset, [87% to 94%].

#### Extensions

This section lists some ideas for extending the tutorial that you may
wish to explore.

- More Algorithms. Only eight machine learning algorithms were evaluated on the
problem; try some linear methods and perhaps some more nonlinear and ensemble methods.

- Algorithm Tuning. No tuning of the machine learning algorithms was performed; mostly
default configurations were used. Pick a method such as SVM, ExtraTrees, or Gradient
Boosting and grid search a suite of different hyperparameter configurations to see if you
can further lift performance on the problem.

- Data Scaling. The data is already scaled to [-1,1], perhaps per subject. Explore whether
additional scaling, such as standardization, can result in better performance, perhaps on
methods sensitive to such scaling such askNN.


#### Further Reading

This section provides more resources on the topic if you are looking to
go deeper.

- scikit-learn: Machine Learning in Python.
http://scikit-learn.org/stable/

- sklearn.metrics.accuracyscoreAPI.
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_
score.html

- sklearn.neighborsAPI.
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors

- sklearn.treeAPI.
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

- sklearn.svmAPI.
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm


- sklearn.naivebayesAPI.
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes

- sklearn.ensembleAPI.
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

#### Summary

In this tutorial, you discovered how to evaluate a diverse suite of
machine learning algorithms

on theActivity Recognition Using Smartphonesdataset. Specifically, you
learned:

- How to load and evaluate nonlinear and ensemble machine learning algorithms on the
feature-engineered version of the activity recognition dataset.

- How to load and evaluate machine learning algorithms on the raw signal data for the
activity recognition dataset.

- How to define reasonable lower and upper bounds on the expected performance of more
sophisticated algorithms capable of feature learning, such as deep learning methods.
