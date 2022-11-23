# 22FALL_ECE143_GROUP13
# Alcohol Effects On Academic Performance
ECE143 (Fall 2022) Course Project by: Linfeng Wen, Liyang Ru, Yuzhao Chen, Amber Szulc, Shril Mody

# Project target
- Analyze the impact of students' drinking habits and other factors on students’ academic performance;
- Use statistical and  ML models to support conclusions and try to find some potential interactions among given variables.

# Future applications
- Improve students' studying achievement is a concern of every school and parents. In addition to learning efficiency in school, drinking habits and family factors also have a great impact on students' learning. This project aims to evaluate the impact of these factors on studying and provide guidance to students.

# Data prepossing and preliminary analysis
## Import Modules

## Installation

Here are the requirements for running all the project codes.
```
pandas==1.5.1
ipython==8.6.0
sklearn==0.0
seaborn==0.12.1
numpy==1.23.5
matplotlib==3.6.0
matplotlib-inline==0.1.6
prettytable==3.5.0
plotly==5.11.0
lazypredict==0.2.12
pydotplus==2.0.2
```
As a recommended practice, you can create virtual environment and install the required packages in the following way
```
# Clone the repo
git clone https://github.com/KeterWen/22FALL_ECE143_GROUP13.git

# Create virtual env
conda create -n alcohol_analysis python=3.8 

# Activate the virual env
conda activate alcohol_analysis

# cd into the root folder and install requirements
pip install requirements.txt
```

To get this section to work, implement the modules located in the Import Modules section of this readme. Please run the main
### ECE143_visualization.ipynb 
file to run all code

## Dataset
- Kaggle: Alcohol Effects on Study (https://www.kaggle.com/datasets/whenamancodes/alcohol-effects-on-study)
- This dataset contains 1044 student exam scores of math and portuguese and their personal information collected by two Portuguese secondary schools. Each piece of data includes 33 different students’ personal information, including alcohol usage condition, student grades, family status (parental education level and job), and living habits (study time, drinking habits, etc.)

## Dataset Analysis and plots
In order to analyze how alcohol and other factors affect academic performance, Exploratory Data Analysis (EDA) will play a critical role in discovering these potential relationships. We will use the histogram, heatmap, and scatterplot to help intuitively present the distribution of different factors including alcohol usage, sex, age, etc. Besides, some charts and quantitative indicators including the Pearson correlation coefficient or mutual information will be leveraged to help measure the correlation between the factors listed above and the final academic performance.

### Distribution
- For both math grades and portuguese grades, shows the histogram about distribution of final grade, workday alcohol consumption and other non-alcohol features which is important.

![](https://github.com/KeterWen/22FALL_ECE143_GROUP13/blob/main/plot/hist_G3.png)
<br>
_Distribution of final grades_

### Heatmap
- In order to show the Pearson's correlations between different comparable features.

![](https://github.com/KeterWen/22FALL_ECE143_GROUP13/blob/main/utils/plotter/plot/maths_heatmap.png)
<br>
_Heatmap of features in math.csv_

### Alcohol Effect on Academic Performance
- Show the relationships between grades and workday/weekend alcohol consumption for analysis the degree of association.

![](https://github.com/KeterWen/22FALL_ECE143_GROUP13/blob/main/utils/plotter/plot/bar_Dalc_vs_G.png)
<br>
_Bar chart of workday alcohol vs. grades_

### Academic improvement
- Exploration between the difference of each grades.

![](https://github.com/KeterWen/22FALL_ECE143_GROUP13/blob/main/plot/math_improvment.png)
<br>
_Math grade improvements_

### Non-Alcohol Effect on Academic Performance
- Show the relationships between other non-alcohol features for analysis the degree of association.


![](https://github.com/KeterWen/22FALL_ECE143_GROUP13/blob/main/utils/plotter/plot/bar_goout_vs_DWalc.png)
<br>
_Bar chart of go out with friends vs. workday alcohol_

# Modeling and Feature Importance

To get this section to work, implement the modules located in the Import Modules section of this readme. Please run the main
### ECE143_visualization.ipynb 
file to run all code

## Determining the Model

- Use lazy classifier to determine most accurate model for predicting data out of set of 29 models
  
![image](https://user-images.githubusercontent.com/91287767/203299756-b7ee25eb-75b8-48ea-99c5-3b2962c95ffd.png)

This ended up being **Random Forest Classifier** and **Random Forest Regressor** for Portuguese Class
and **Decision Tree Classifier** and **Decision Tree Regressor** for Math Class.

Decision Tree and Random Forest were used in the code ahead for Math and Portuguese respectively.

## Using the Model to Predict Grades

- We use both models chosen in order to predict the test set for grades 3 (G3) in order to determine how accurate our model is. 

- Understanding how accurate our model is will give us an understanding about the errors in accuracy for feature importance.

- We plotted a visual comparison between predicted grades and actual, a plot of their difference, and a plot of the error rate

![image](https://user-images.githubusercontent.com/91287767/203300114-9c27e22d-d2b9-475c-81f3-f42c18ac319a.png)


## Visual Representation of the Models

- To provide the user an understanding of our models, we gave a visual diagram of the trees trained by our model to show how the models worked in simpler form. (Trees shown only with a depth of 3)

![image](https://user-images.githubusercontent.com/91287767/203300044-06bb2cb6-70d0-4563-8acc-de6703051947.png)


## Feature Importance

- Using the 2 most accurate models, we determined which out of the 30 features were most important in determining grades. 

- We plotted a bar graph that showed how much each feature was important in determining the grades. We sorted these features based on their impact on grades in descinding order. 

_(Left - Most Impact, Right - Least Impact)_
![image](https://user-images.githubusercontent.com/91287767/203299314-687b4aea-ce1f-414e-b825-db4c13506ebb.png)


## Feature Importance Table

- These tables shown below give conclusion about what were the most important factors in determining grades for math and portuguese class. 

- It shows how alchohol consumption ranked in influencing grades out of all factors and how much (% wise) alcohol consumption contributed to final grades.

_Table for Portuguese (Random Forest Classifier)_

![image](https://user-images.githubusercontent.com/91287767/203299197-c8d6dea1-6970-4e3f-8198-69b2bb18c38c.png)

_Table for Math (Decision Tree Classifier)_

![image](https://user-images.githubusercontent.com/91287767/203299253-b19f44dc-db57-4c0d-a864-00d4c707dffd.png)




