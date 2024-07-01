Titanic Survival Prediction
This script uses a Random Forest Classifier to predict the survival of passengers on the Titanic based on their demographic and travel information. The data is sourced from the well-known Titanic dataset.

->Prerequisites
->Python 3.x
->Pandas library
->Scikit-learn library

Script:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Load the data
train_data = pd.read_csv(r"C:\dbms\train (1).csv")
test_data = pd.read_csv(r"C:\dbms\test.csv")

# Display the first few rows of the datasets
print(train_data.head())
print(test_data.head())

# Calculate the survival rates for women and men
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("% of men who survived:", rate_men)

# Prepare the data for the model
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Create the output DataFrame and save to CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

Description:
This script performs the following tasks:

Import Libraries:
Imports necessary libraries for data processing (pandas, numpy) and machine learning (RandomForestClassifier from scikit-learn).
Load Data:

Loads training and testing data from specified file paths.
Display Data:

Prints the first few rows of the datasets to inspect the data structure.
Calculate Survival Rates:

Calculates and prints the survival rates of women and men in the training dataset.
Prepare Data for Modeling:

Extracts the target variable Survived from the training data.
Selects features (Pclass, Sex, SibSp, Parch) and applies one-hot encoding using pd.get_dummies to convert categorical variables into numeric format.
Train the Model:

Initializes a RandomForestClassifier with 100 trees and a maximum depth of 5.
Trains the model on the training data.
Make Predictions:

Uses the trained model to predict survival on the test data.
Save Predictions:

Creates a DataFrame with PassengerId and Survived columns for the test data predictions.
Saves the predictions to a CSV file named submission.csv.

Output:
Running this script will produce a CSV file named submission.csv containing the predicted survival outcomes for the test data. The file will be saved in the current working directory.

   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
   PassengerId  Pclass                                          Name     Sex  \
0          892       3                              Kelly, Mr. James    male   
1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   
2          894       2                     Myles, Mr. Thomas Francis    male   
3          895       3                              Wirz, Mr. Albert    male   
4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   

    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  
0  34.5      0      0   330911   7.8292   NaN        Q  
1  47.0      1      0   363272   7.0000   NaN        S  
2  62.0      0      0   240276   9.6875   NaN        Q  
3  27.0      0      0   315154   8.6625   NaN        S  
4  22.0      1      1  3101298  12.2875   NaN        S  
% of women who survived: 0.7420382165605095
% of men who survived: 0.18890814558058924
Your submission was successfully saved!
