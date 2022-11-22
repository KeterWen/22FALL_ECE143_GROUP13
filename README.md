# 22FALL_ECE143_GROUP13


# Modeling and Feature And Importance
## Determining the Model
Use lazy classifier to determine most accurate model for predicting data out of set of 29 models
This ended up being Random Forest Classifier and Random Forest Regressor for Portuguese aClass
and Decision Tree Classifier and Decision Tree Regressor for Math Class 

## Using the Model to Predict Grades
We use both models chosen in order to predict the test set for grades 3 in order to determine how accurate our model is. 
Understanding how accurate our model is will give us an understanding about the errors in accuracy for feature importance
We plotted a visual comparison between predicted grades and actual, a plot of their difference, and a plot of the error rate


## Visual Representation of the Models
To provide the user / reader an understanding of our models we used, we gave a visual diagram (the trees) to show how the models worked in simpler form (only with a depth of 3 to save time)


## Feature Importance
Using the 2 most accurate models, we determined which out of the 30 features were most important to least in determining grades. 
We plotted a bar graph that showed how much each feature was important in determining grades. We sorted these features from most impact on grades (left) to least right)

## Feature Importance Table
These tables printed gave the conclusion about what the most important factors in determining grades for math and portuguese class were. It showed how 
alchohol consumption ranked in influencing grades out of all factors and how much (percentage wise) alcohol consumption contributed to final grades


Please import the following modules to get the modeling feature importance to work:
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from prettytable import PrettyTable
from six import StringIO 
from IPython.display import Image  
from sklearn.tree import plot_tree

import pydotplus
