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

