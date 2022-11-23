# 22FALL_ECE143_GROUP13
# Alcohol Effects On Academic Performance

# Modeling and Feature And Importance

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



## Import Modules

Please import the following modules to get the modeling feature importance to work:

![image](https://user-images.githubusercontent.com/91287767/203297270-d206e2c3-66db-4a51-9864-4876d05064f1.png)

To install the following modules, run the following code - 
```
pip install -r requirements.txt
```

