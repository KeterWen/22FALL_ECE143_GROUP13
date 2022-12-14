#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lazypredict.Supervised import LazyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from prettytable import PrettyTable
from sklearn.tree import plot_tree


def Ctgrs_to_nums(my_subject):
    '''
    Turn dataset categories into numbers
    :param my_subject: dataset of all student features and grades per class 
    :return:
        my_subject: dataset of all student features and grades per class
    '''
    #convert categories into numbers to allow for modeling
    LEncode = LabelEncoder()
    each_feature = my_subject.select_dtypes(include = ['object']).columns
    for column in each_feature:
        my_subject[column] = LEncode.fit_transform(my_subject[column])
    return my_subject

def train_data(my_subject):
    '''
    Train datasets to predict future grades based off current information
    :param my_subject: dataset of all student features and grades per class
    :return:
        my_subject_x_test_t: unedited all features of the students used to predict preformance, training set input
        my_subject_x_test: all features of the students used to predict preformance, training set input
        my_subject_y_train: The training set output, G# grades
        my_subject_y_test: The test set output ie G# grades
        my_subject_x_train: all 30+ feature training set to determine grades
    '''
    my_subject_x = my_subject.copy();
    my_subject_x.drop('G3',axis = 1,inplace = True)
    my_subject_y = my_subject['G3']
    my_subject_x_train_t, my_subject_x_test, my_subject_y_train, my_subject_y_test = train_test_split(my_subject_x, my_subject_y, test_size=0.2,random_state=0)
    standScale = StandardScaler()
    my_subject_x_train = standScale.fit_transform(my_subject_x_train_t)
    my_subject_x_test = standScale.transform(my_subject_x_test)
    return my_subject_x_train_t, my_subject_x_test, my_subject_y_train, my_subject_y_test, my_subject_x_train
    
def lzy_clsfy_models(my_subject_x_test, my_subject_y_train, my_subject_y_test, my_subject_x_train):
    '''
    Create a lazy classifier to determine which model is best
    :param my_subject_x_test: all features of the students used to predict preformance, training set input
    :param my_subject_y_train: The training set output, G# grades
    :param my_subject_y_test: The test set output ie G# grades
    :param my_subject_x_train: all 30+ feature training set to determine grades
    :return: model_my_subject: dataset of all student features and grades per class model of training data
    '''
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None, classifiers = 'all')
    
    model_my_subject,prediction_my_subject = clf.fit(my_subject_x_train, my_subject_x_test, my_subject_y_train, my_subject_y_test)
    return model_my_subject


def plt_models(model_subject, my_subject_name='Math'):
    '''
    Plot accuracy of all possible models that evaluate dataset
    :param model_subject: name of the model
    :param my_subject_name: The name of the school subject students are taking
    :return:none
    '''
    model_subject.sort_values(by = 'Accuracy',inplace = True,ascending = False)
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=model_subject.index, y="Accuracy", data=model_subject)
    plt.xticks(rotation=90)
    plt.title('Accuracy for %s Model' %my_subject_name)
    plt.show()


def train_data_no_grades(my_subject):
    '''
    Train features to determine future grades without using prior grades as a feature
    :param my_subject: dataset of all student features and grades per class
    :return:
        my_subject_x_test_t: unedited all features of the students used to predict preformance, training set input
         my_subject_x_test: all features of the students used to predict preformance, training set input
         my_subject_y_train: The training set output, G# grades
        my_subject_y_test: The test set output ie G# grades
        my_subject_x_train: all 30+ feature training set to determine grades
    '''
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


#take the most successful prediction model and compare predicted data versus actual data
# take the most successful prediction model and compare predicted data versus actual data
def predict_grades(my_subject_name, my_subject_x_test, my_subject_y_train, \
                   my_subject_y_test, my_subject_x_train, inst = 0):
    '''
    Predict grades using a model
    :param my_subject_name: The name of the school subject students are taking
    :param my_subject_x_test: all features of the students used to predict preformance, training set input
    :param my_subject_y_train: The training set output, G# grades
    :param my_subject_y_test: The test set output ie G# grades
    :param my_subject_x_train: all 30+ feature training set to determine grades
    :param instance of which model the program is on
    :return:none
    '''
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
    '''
    Plot predicted versus actual grades for visual comparison
    :param predictions: possible output using model of the data
    :param my_subject_y_test: The test set output ie G# grades
    :param subject: name of dataset of all student features and grades per class 
    :param model: name of the model in use
    :return:none
    '''
    plt.plot(range(len(predictions)), predictions, label ="")
    plt.plot(range(len(predictions)), my_subject_y_test, label ="")
    plt.title("%s:" % subject               +" %s\'s" % model +" Predicted Grades Compared Against Actual Grades")
    plt.xlabel("Random Test Set #")
    plt.ylabel("Grades on scale from 1 to 20")
    plt.legend(["Predicted Grades", "Actual Grades"], loc ="lower right")
    plt.show()
def plot_pred_v_actual_diff(predictions, my_subject_y_test, subject, model):
    '''
    Plot the difference between actual grades and the predicted grade values
    :param predictions: possible output using model of the data
    :param my_subject_y_test: The test set output ie G# grades
    :param subject: name of dataset of all student features and grades per class 
    :param model: name of the model in use
    :return:none
    '''
    plt.plot(range(len(predictions)), abs(my_subject_y_test-predictions))
    plt.title("%s:" % subject               +" %s\'s" % model + "Predicted Grades Compared Against Actual Grades")
    plt.xlabel("Random Test Set #")
    plt.ylabel("Grades Difference in Predicted Grades vs Actual Grades")
    # zero means perfect correlation
    plt.show()
# plot predictions, and the y_tEst OveR my_subject_x_tesT to see how well prediction preformed
def plot_error_rate(predictions, my_subject_y_test, subject, model):
    '''
    Plot error of how far off predicted grades were from actual
    :param predictions: possible output using model of the data
    :param my_subject_y_test: The test set output ie G# grades
    :param subject: name of dataset of all student features and grades per class 
    :param model: name of the model in use
    :return: none
    '''
    # Error Rate = |Observed Value - Actual Value|/Actual Value ?? 100
    plt.bar(range(len(predictions)), abs(predictions-my_subject_y_test)/my_subject_y_test *100 )
    plt.title("%s:" % subject               +" %s\'s" % model + "Error Rate for Grades Predicted vs actual")
    plt.xlabel("Random Test Set #")
    plt.ylabel("Error Rate Percentage")
    # zero means perfect correlation
    plt.show()


def determine_tree(inst = 0):
    '''
    Determines which model to use
    :param inst: instance of which model the program is on
    :return:
        tree: model for interpreting and predicting data
        model_name: Name of model for interpreting and predicting data
    '''
    if inst == 0:
        tree = DecisionTreeRegressor()
        model_name = "Decision Tree Regressor"
    else:
        tree = RandomForestRegressor()
        model_name = "Random Forest Regressor"
    return tree, model_name

def visualize_tree(my_subject_x_train, my_subject_y_train, inst = 0):
    '''
    Visualize the different models we used to predict feature importance
    :param my_subject_x_train: all 30+ feature training set to determine grades
    :param my_subject_y_train: The training set output, G# grades
    :param instance of which model the program is on
    :return: none
    '''
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
    '''
    Finds and ranks which features are important on a plot
    :param tree: the function to interpret data into a model
    :param my_subject_x: all 30+ features dataset
    :param my_subject_x_train: all 30+ feature training set to determine grades
    :param my_subject_y_train: The training set output, G# grades
    :return:
        names: the names of the features  (names of columns in dataset)
        values: the quantitative data of the features for each student
        sorted_feature_importance: dictionary staring from most important features to least
    '''
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
    '''
    Plot how important each feature is and how much it contrivutes to overall grade output
    :param names: the names of the features  (names of columns in dataset)
    :param values: the quantitative data of the features for each student
    :param my_subject_name: The name of the school subject students are taking
    :param model: name of the model in use name of the model in use
    : return: none
    '''


    plt.bar(x=names, height=values, color='#087E8B')
    plt.suptitle('%s Grades:' % my_subject_name   \
        +' Feature Importance \n How important'  \
        + 'each feature is in determining %s' % my_subject_name   \
                 + 'Grade 3 \n using %s' % model, size = 10)
    plt.xticks(rotation='vertical')
    plt.show()


def table_feature_imptnc(sorted_feature_importance, names):
    '''
    Create table of the most important features and rank alcohol among features
    :param sorted_feature_importance: dictionary staring from most important features to least
    :param names: the names of the features  (names of columns in dataset)
    : return: none
    '''
    FeatureTable = PrettyTable(["The 5 most important feautures"])
    FeatureTable.add_row([str(list(sorted_feature_importance.keys())[:5])])
    print(FeatureTable)
    rankWalc = names.index('Walc') + 1
    WalcImportance =  str("%.2f" % (sorted_feature_importance.get('Walc')*100))
    rankDalc = list(sorted_feature_importance).index('Dalc') + 1                     
    DalcImportance =  str("%.2f" % (sorted_feature_importance.get('Dalc') *100))
    print("Weekend Alchohol consumption ranks #" + str(rankWalc) + " and determines "      + WalcImportance + " Percent of the overall grade")
    print("WeekDAY Alchohol consumption ranks #" + str(rankDalc) + " and determines "      + DalcImportance + " Percent of the overall grade")


def model_and_feature2(mysubject, my_subject_name):
    '''
    Calls host of functions to run the model, Uses trhe model to predict grades, visualizes model
    uses model to determine feature importance, plot feature importance, table to rank feature importance
    :param mysubject: the dataframe of the dataset for students' differing life factors
    :param my_subject_name: The name of the school subject students are taking
    :return: none
    '''
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

if __name__ == "__main__":
    pass
    # math = data_utils.read_csv('Maths.csv')
    # portuguese = data_utils.read_csv('Portuguese.csv')
    # model_and_feature2(math, 'Math')
     #model_and_feature2(portuguese, 'Portuguese')