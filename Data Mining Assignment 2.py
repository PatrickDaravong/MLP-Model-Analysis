#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:44:34 2023

@author: patrickdaravong
"""

'STEP 1A IMPORT THE DATA'

import pandas as pd
data = pd.read_csv("/Users/patrickdaravong/Downloads/Car_Data.csv")

"Using data.info we can check for what the null values are"
data.info()

"We can see that person has the most null with 1152 entries so we use value_count to see the count of the individual entries"
Person_count= data['Person'].value_counts()
Person_count

"We see that person has an equal amount of 4 and 2, we cannot impute mean as Person only takes specific value so we will equally impute 4 and 2 "

"Get the exact number missing"
num_person_null = data['Person'].isna().sum()

"Impute missing values with either 2 or 4 "
num_to_impute_person = num_person_null // 2

"Impute equal amounts of 2s and 4s"
data['Person'].fillna(2, limit=num_to_impute_person, inplace=True)
data['Person'].fillna(4, limit=num_to_impute_person, inplace=True)

"The next most nulls is Boot with 115 null entries so we will again count the entries"
Boot_count = data['Boot'].value_counts()
Boot_count

"Boot has three entries, small, med, big and Like person we can see there are equal amounts of each so will impute equal amounts of each"

"Get the exact number missing"
num_boot_null = data['Boot'].isna().sum()

num_to_impute_boot = num_boot_null //3

"Impute equal amounts of small, med, big"
data['Boot'].fillna('small', limit = num_to_impute_boot, inplace= True)
data['Boot'].fillna('med', limit = num_to_impute_boot, inplace= True)
data['Boot'].fillna('big', limit = num_to_impute_boot, inplace= True)

"The next most nulls is maint_cost with 41 missing entries so at this point we can just remove any null entries as they are not a large portion of our dataset "
data.dropna(inplace=True)

"Check with data info"
data.info()

"Next we have to perform some Data Encoding as when building a NN we have to make sure all our values are Numerical"
"Create custom encoding Mapping: BUY & Maint_cost"

"Since our values are ordinal we are going to go from 0 to desired number"
buy_mapping = {'low': 0, 'med': 1, 'high': 2,'vhigh':3}

"Apply Mapping to Buy and Maint_cost"
data['Buy'] = data['Buy'].map(buy_mapping)

data['Maint_costs'] = data['Maint_costs'].map(buy_mapping)

"Replace 5more for doors so all values are numerical"
data['Doors'].replace("5more", 5, inplace=True)

"Apply mapping to Lug boot"
lug_mapping = {"small":0,"med":1,"big":2}

data["Boot"] = data["Boot"].map(lug_mapping)

"Apply mapping to Safety "
Safety_mapping = {'low': 0, 'med': 1, 'high': 2}

data["Safety"] = data["Safety"].map(Safety_mapping)

"Apply mapping to Quality"

quality_mapping = {"unacc": 0,"acc": 1,"good": 2,"vgood": 3}

data["Quality"] = data["Quality"].map(quality_mapping)


"We now export the csv with our preprocessing"
filepath = '/Users/patrickdaravong/Downloads/output.csv'
data.to_csv(filepath, index=False)

data = pd.read_csv('/Users/patrickdaravong/Downloads/output.csv')


'STEP 2 CREATE NN'
'Split the training set 25:75'
from sklearn.model_selection import train_test_split

array = data.values
X = array[:,0:6]
y = array[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

'Create and Train the model'
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,log_loss,roc_auc_score, recall_score


"Model 1 - Base line model everything is default"
mlp = MLPClassifier(
             random_state=123) 

mlp.fit(X_train, y_train) # Train the algorithm
predictions = mlp.predict(X_test) # Make predictions

"Evaluate results"

"Loss"
mlp_pred = mlp.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred))

"Cost"
print(confusion_matrix(y_test, predictions))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred,multi_class='ovr'))



"Model 2 - We change hidden layer to 10"
mlp2 = MLPClassifier(hidden_layer_sizes=(10),
                    random_state=123) 

mlp2.fit(X_train, y_train) # Train the algorithm
predictions2 = mlp2.predict(X_test) # Make predictions

"Loss"
mlp_pred2 = mlp2.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred2))

"Cost"
print(confusion_matrix(y_test, predictions2))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions2))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions2,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions2, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred2,multi_class='ovr'))

"Model 3 - we change hidden layer to 1000"
mlp3 = MLPClassifier(hidden_layer_sizes=(1000),
                    random_state=123) 

mlp3.fit(X_train, y_train) # Train the algorithm
predictions3 = mlp3.predict(X_test) # Make predictions

"Loss"
mlp_pred3 = mlp3.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred3))

"Cost"
print(confusion_matrix(y_test, predictions3))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions3))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions3,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions3, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred3,multi_class='ovr'))


"Model 4 - we now add the max_iter parameter and start it at 10"
mlp4 = MLPClassifier(hidden_layer_sizes=(1000),
                    max_iter=10,
                    random_state=123) 

mlp4.fit(X_train, y_train) # Train the algorithm
predictions4 = mlp4.predict(X_test) # Make predictions

"Loss"
mlp_pred4 = mlp4.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred4))

"Cost"
print(confusion_matrix(y_test, predictions4))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions4))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions4,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions4, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred4,multi_class='ovr'))


"Model 5 - we change the iter to 20000"
mlp5 = MLPClassifier(hidden_layer_sizes=(1000),
                    max_iter=20000,
                    random_state=123) 

mlp5.fit(X_train, y_train) # Train the algorithm
predictions5 = mlp5.predict(X_test) # Make predictions

"Loss"
mlp_pred5 = mlp5.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred5))

"Cost"
print(confusion_matrix(y_test, predictions5))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions5))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions5,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions5, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred5,multi_class='ovr'))


"Model 6 - we add learning_rate as a fine tuning paramter and use invscaling"
mlp6 = MLPClassifier(hidden_layer_sizes=(100),
                    max_iter=200,
                    solver='sgd',
                    learning_rate= 'invscaling',
                    random_state=123) 

mlp6.fit(X_train, y_train) # Train the algorithm
predictions6 = mlp6.predict(X_test) # Make predictions

"Loss"
mlp_pred6 = mlp6.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred6))

"Cost"
print(confusion_matrix(y_test, predictions6))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions6))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions6,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions6, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred6,multi_class='ovr'))

'Model 7 - we change the learning_rate to adaptive'
mlp7 = MLPClassifier(hidden_layer_sizes=(100),
                    max_iter=200,
                    solver='sgd',
                    learning_rate= 'adaptive',
                    random_state=123) 

mlp7.fit(X_train, y_train) # Train the algorithm
predictions7 = mlp7.predict(X_test) # Make predictions

"Loss"
mlp_pred7 = mlp7.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred7))

"Cost"
print(confusion_matrix(y_test, predictions7))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions7))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions7,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions7, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred7,multi_class='ovr'))

'Model 8 - we now add the optomiser fine tuner and use sgd'
mlp8 = MLPClassifier(hidden_layer_sizes=(100),
                    max_iter=20000,
                    solver='sgd',
                    random_state=123) 

mlp8.fit(X_train, y_train) # Train the algorithm
predictions8 = mlp8.predict(X_test) # Make predictions

"Loss"
mlp_pred8 = mlp8.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred8))

"Cost"
print(confusion_matrix(y_test, predictions8))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions8))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions8,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions8, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred8,multi_class='ovr'))


'Model 9 - we use the lbfgs solver'
mlp9 = MLPClassifier(hidden_layer_sizes=(100),
                    max_iter=20000,
                    solver='lbfgs',
                    random_state=123) 

mlp9.fit(X_train, y_train) # Train the algorithm
predictions9 = mlp9.predict(X_test) # Make predictions

"Loss"
mlp_pred9 = mlp9.predict_proba(X_test)
print("Loss =",log_loss(y_true = y_test, y_pred = mlp_pred9))

"Cost"
print(confusion_matrix(y_test, predictions9))

"Accuracy"
print("Accuracy =", accuracy_score(y_test, predictions9))

"Sensitivity"
print("Sensitivity =",recall_score(y_test,predictions9,average = None))

"Unweighted Average Recall"
print("UAR =", recall_score(y_test, predictions9, average= 'macro'))

"AUC"
print("AUC =", roc_auc_score(y_test,mlp_pred9,multi_class='ovr'))


"We now plot the best curve"
import matplotlib.pyplot as plt

lossPerIteration = mlp.loss_curve_ #model loss curve
plt.figure(dpi=600)
plt.plot(lossPerIteration)
plt.title("Model 1 Loss curve")
plt.xlabel("The Number of iterations")
plt.ylabel("The Training Loss")
plt.show()









