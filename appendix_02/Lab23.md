<img align="right" src="../logo-small.png">


# How to Setup a Workstation for Deep Learning

It can be difficult to install a Python machine learning environment on some platforms. Python
itself must be installed first and then there are many packages to install, and it can be confusing
for beginners. In this tutorial, you will discover how to setup a Python machine learning
development environment.

After completing this tutorial, you will have a working Python environment to begin learning,
practicing, and developing machine learning and deep learning softwares.

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-for-nlp` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab19_Setup`

**Note:** Terminal is already running. You can also open new terminal by clicking:
`File` > `New` > `Terminal`.

To copy and paste: use **Control-C** and to paste inside of a terminal, use **Control-V**

#### Python
In this environment you can check that Python is already installed by running `python3 --version`

To locally develop and run Python code, it is recommended to use a Python virtual environment. Run the following commands to create and activate a virtual environment named `.venv`.

`apt-get update`

`apt-get install -y python3-venv`

`python3 -m venv .venv`

`source .venv/bin/activate`

In the terminal with the virtual environment activated, run the following command in the start folder to install the dependencies. Some installation steps may take a few minutes to complete.

`pip install --upgrade pip`

`pip install scipy`

`pip install numpy`

`pip install matplotlib`

`pip install pandas`

`pip install statsmodels`

`pip install sklearn`

The script below will print the version number of the key SciPy libraries you require for
machine learning development, specifically: SciPy, NumPy, Matplotlib, Pandas, Statsmodels,
and Scikit-learn. You can type python and type the commands in directly. Alternatively, I
recommend opening a text editor and copy-pasting the script into your editor.

```
# check library version numbers
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```

##### Run Notebook
Click notebook `python_versions.ipynb` in jupterLab UI and run jupyter notebook.

You should see output like the following:

```
scipy: 1.2.1
numpy: 1.16.2
matplotlib: 3.0.3
pandas: 0.24.2
statsmodels: 0.9.0
sklearn: 0.20.3
```

In this step, we will install Python libraries used for deep learning, specifically: Theano,
TensorFlow, and Keras. **Note:** I recommend using Keras for deep learning and Keras only
requires one of Theano or TensorFlow to be installed. You do not need both. There may be
problems installing TensorFlow on some Windows machines.

1. Install the Theano deep learning library by typing:
`pip install theano` 

2. Install the TensorFlow deep learning library by typing:
`pip install tensorflow` 

3. Install Keras by typing:
`pip install keras` 

Confirm your deep learning environment is installed and working correctly. Create a script that prints the version numbers of each library, as we did before for the SciPy
environment.

```
# check deep learning version numbers
# theano
import theano
print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)
```

##### Run Notebook
Click notebook `deep_versions.ipynb` in jupterLab UI and run jupyter notebook.

You should see output like the following:

```
theano: 1.0.4
tensorflow: 1.13.1
keras: 2.2.4
```

#### Summary
Congratulations, you now have a working Python development environment for machine learning
and deep learning. You can now learn and practice machine learning and deep learning on your
workstation.

