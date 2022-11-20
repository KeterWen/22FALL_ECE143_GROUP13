#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sys
get_ipython().system('{sys.executable} -m pip install lazypredict')
get_ipython().system('{sys.executable} -m pip install plotly.express')
get_ipython().system('{sys.executable} -m pip install prettytable')
get_ipython().system('{sys.executable} -m pip install six')
get_ipython().system('{sys.executable} -m pip install pydotplus')
get_ipython().system('{sys.executable} -m pip install graphviz')


# In[41]:



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




def my_read_csv(filename):
    '''
    collect the csv file data
    '''
    subject = pd.read_csv(filename,index_col=None)
    return subject

def Ctgrs_to_nums(my_subject):
    #convert categories into numbers to allow for modeling
    LEncode = LabelEncoder()
    each_feature = my_subject.select_dtypes(include = ['object']).columns
    for column in each_feature:
        my_subject[column] = LEncode.fit_transform(my_subject[column])
    return my_subject


# In[6]:


def train_data(my_subject):
    my_subject_x = my_subject.copy();
    my_subject_x.drop('G3',axis = 1,inplace = True)
    my_subject_y = my_subject['G3']
    my_subject_x_train_t, my_subject_x_test, my_subject_y_train, my_subject_y_test = train_test_split(my_subject_x, my_subject_y, test_size=0.2,random_state=0)
    standScale = StandardScaler()
    my_subject_x_train = standScale.fit_transform(my_subject_x_train_t)
    my_subject_x_test = standScale.transform(my_subject_x_test)
    return my_subject_x_train_t, my_subject_x_test, my_subject_y_train, my_subject_y_test, my_subject_x_train
    
def lzy_clsfy_models(my_subject_x_test, my_subject_y_train, my_subject_y_test, my_subject_x_train):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None,classifiers = 'all')
    model_my_subject,prediction_my_subject = clf.fit(my_subject_x_train, my_subject_x_test, my_subject_y_train, my_subject_y_test)
    return model_my_subject


# In[7]:


def plt_models(model_subject, my_subject_name='Math'):
    model_subject.sort_values(by = 'Accuracy',inplace = True,ascending = False)
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=model_subject.index, y="Accuracy", data=model_subject)
    plt.xticks(rotation=90)
    plt.title('Accuracy for %s Model' %my_subject_name)
    plt.show()


# In[32]:


def train_data_no_grades(my_subject):
    my_subject_x = my_subject.copy();
    my_subject_x.drop('G3', axis=1, inplace=True)
    my_subject_y = my_subject['G3']
    my_subject_x.drop('G1', axis=1, inplace=True)
    my_subject_x.drop('G2', axis=1, inplace=True)
    my_subject_x_train_t, my_subject_x_test, my_subject_y_train, my_subject_y_test =    train_test_split(my_subject_x, my_subject_y, test_size=0.2, random_state=0)
    standScale = StandardScaler()
    my_subject_x_train = standScale.fit_transform(my_subject_x_train_t)
    my_subject_x_test = standScale.transform(my_subject_x_test)
    return my_subject_x_train_t, my_subject_x_test, my_subject_y_train,         my_subject_y_test, my_subject_x_train, my_subject_x, my_subject_y


# In[51]:


#take the most successful prediction model and compare predicted data versus actual data
# take the most successful prediction model and compare predicted data versus actual data
def predict_grades(my_subject_name, my_subject_x_test, my_subject_y_train,                    my_subject_y_test, my_subject_x_train, inst = 0):
    if inst == 0:
        clf = RandomForestClassifier(random_state=0)
        model_name = "Random Forest Classifier"
    else:
        clf = DecisionTreeClassifier()
        model_name = "Decision Tree Classifier"
    clf.fit(my_subject_x_train, my_subject_y_train)
    predictions = clf.predict(my_subject_x_test)
    return predictions, model_name

def plot_pred_v_actual(predictions, my_subject_y_test, subject, model):
    plt.plot(range(len(predictions)), predictions, label ="")
    plt.plot(range(len(predictions)), my_subject_y_test, label ="")
    plt.title("%s:" % subject               +" %s\'s" % model +" Predicted Grades Compared Against Actual Grades")
    plt.xlabel("Random Test Set #")
    plt.ylabel("Grades on scale from 1 to 20")
    plt.legend(["Predicted Grades", "Actual Grades"], loc ="lower right")
    plt.show()
def plot_pred_v_actual_diff(predictions, my_subject_y_test, subject, model):
    plt.plot(range(len(predictions)), abs(my_subject_y_test-predictions))
    plt.title("%s:" % subject               +" %s\'s" % model + "Predicted Grades Compared Against Actual Grades")
    plt.xlabel("Random Test Set #")
    plt.ylabel("Grades Difference in Predicted Grades vs Actual Grades")
    # zero means perfect correlation
    plt.show()
# plot predictions, and the y_tEst OveR my_subject_x_tesT to see how well prediction preformed
def plot_error_rate(predictions, my_subject_y_test, subject, model):
    # Error Rate = |Observed Value - Actual Value|/Actual Value × 100
    plt.bar(range(len(predictions)), abs(predictions-my_subject_y_test)/my_subject_y_test *100 )
    plt.title("%s:" % subject               +" %s\'s" % model + "Error Rate for Grades Predicted vs actual")
    plt.xlabel("Random Test Set #")
    plt.ylabel("Error Rate Percentage")
    # zero means perfect correlation
    plt.show()


# In[68]:


def determine_tree(inst = 0):
    if inst == 0:
        tree = DecisionTreeRegressor()
        model_name = "Decision Tree Regressor"
    else:
        tree = RandomForestRegressor()
        model_name = "Random Forest Regressor"
    return tree, model_name

def visualize_tree(my_subject_x_train, my_subject_y_train, inst = 0):
    if inst == 0:
        DTC = RandomForestRegressor(max_depth = 3)
        print("Random Forest Regression \n")
    else:
        DTC = DecisionTreeRegressor(max_depth = 3)
        print("Decision Tree Regression\n")
    model = DTC.fit(my_subject_x_train, my_subject_y_train)
    print("Note this is only for visual understanding purposes with depth of 3. \n"+    "We do not show entire depth for time saving purposes")
    plt.figure(figsize=(25,25))
    plot_tree(DTC, 
              filled=True, 
              rounded=True, 
              fontsize=14)
    plt.show()
def find_feature_imptnc(tree, my_subject_x, my_subject_x_train, my_subject_y_train):
    features = my_subject_x.columns.tolist()
    # paramters for decision tree regressor

    # Fitting on the training data
    tree.fit(my_subject_x_train, my_subject_y_train)
    importances_sk = tree.feature_importances_
    feature_importance_sk = {}
    for i, feature in enumerate(features):
        feature_importance_sk[feature] = round(importances_sk[i], 3)
    feature_importance_keys = sorted(feature_importance_sk, key=feature_importance_sk.get, reverse=True)

    sorted_feature_importance = {}

    for w in feature_importance_keys:
        sorted_feature_importance[w] = feature_importance_sk[w]

    print(f"Feature importance by sklearn: {feature_importance_sk}")
    names = list(sorted_feature_importance.keys())
    values = list(sorted_feature_importance.values())
    return names, values, sorted_feature_importance

def plot_feature_imptnc(names, values, my_subject_name='Portuguese', model='Random Forest Regressor'):
    plt.bar(x=names, height=values, color='#087E8B')
    plt.suptitle('%s Grades:' % my_subject_name     +' Feature Importance \n How important'      + 'each feature is in determining %s' % my_subject_name      + 'Grade 3 \n using %s' % model, size = 10)
    plt.xticks(rotation='vertical')
    plt.show()


# In[12]:


def table_feature_imptnc(sorted_feature_importance, names):
    FeatureTable = PrettyTable(["The 5 most important feautures"])
    FeatureTable.add_row([str(list(sorted_feature_importance.keys())[:5])])
    print(FeatureTable)
    rankWalc = names.index('Walc') + 1
    WalcImportance =  str("%.2f" % (sorted_feature_importance.get('Walc')*100))
    rankDalc = list(sorted_feature_importance).index('Dalc') + 1                     
    DalcImportance =  str("%.2f" % (sorted_feature_importance.get('Dalc') *100))
    print("Weekend Alchohol consumption ranks #" + str(rankWalc) + " and determines "      + WalcImportance + " Percent of the overall grade")
    print("WeekDAY Alchohol consumption ranks #" + str(rankDalc) + " and determines "      + DalcImportance + " Percent of the overall grade")


# In[69]:


def model_and_feature(mysubject, my_subject_name):
    my_subject = Ctgrs_to_nums(mysubject)
    my_subject_x_train_t, my_subject_x_test, my_subject_y_train,     my_subject_y_test, my_subject_x_train = train_data(my_subject)
    model_my_subject = lzy_clsfy_models(my_subject_x_test, my_subject_y_train, my_subject_y_test, my_subject_x_train)
    plt_models(model_my_subject, my_subject_name)
    my_subject_x_train_t, my_subject_x_test, my_subject_y_train, my_subject_y_test,    my_subject_x_train, my_subject_x, my_subject_y,         = train_data_no_grades(
        my_subject)
    num_of_models= 2
    for inst in range(num_of_models):
        # modeling portion
        predictions, my_model_name = predict_grades(my_subject_name, my_subject_x_test, my_subject_y_train,
                                                    my_subject_y_test, my_subject_x_train, inst)
        plot_pred_v_actual(predictions, my_subject_y_test, my_subject_name, my_model_name)
        plot_pred_v_actual_diff(predictions, my_subject_y_test, my_subject_name, my_model_name)
        plot_error_rate(predictions, my_subject_y_test, my_subject_name, my_model_name)
    for inst in range(num_of_models): 
        tree, my_model_name = determine_tree(inst)
        visualize_tree(my_subject_x_train, my_subject_y_train, inst)
        # feature importance portion
        names, values, sorted_feature_importance = find_feature_imptnc(tree, my_subject_x, my_subject_x_train, my_subject_y_train)
        plot_feature_imptnc(names, values, my_subject_name, my_model_name)
        table_feature_imptnc(sorted_feature_importance, names)


math = my_read_csv('Maths.csv')
portuguese = my_read_csv('Portuguese.csv')
model_and_feature(math, 'Math')
model_and_feature(portuguese, 'Portuguese')


# In[ ]:





# In[ ]:




