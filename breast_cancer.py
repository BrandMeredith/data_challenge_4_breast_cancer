
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:43:41 2019

@author: Brandon

Notes found at: https://docs.google.com/document/d/1QJyLzZbENTAz5VXhm8voFvSeilomcrrYM1ONR3paTOc/edit
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# read in data
df = pd.read_csv('BreastCancerDetection/breast-cancer-wisconsin.txt',index_col=0)

# Clean the data
##########################################################
# coerce strings to ints or NaNs
for column in df.columns[1:]:
    df[column] = pd.to_numeric(df[column],downcast='integer',errors='coerce')

# drop NaNs
df = df.dropna()

# drop data that's out of range
for column in df.columns[1:]:
    indices = df[(df[column]<=0) |( df[column]>=11)].index
    df = df.drop(labels=indices,axis=0)  

# map benign to 0 and malignant to 1
df.Class = df.Class.map({2:0,4:1})
###########################################################

df.head()
df.info()
df.describe()

# plot the data
columns = df.columns
for column in columns:
    plt.figure()
    plt.scatter(df[column],df['Class'])

# plot correlation matrix
plt.figure()
plt.matshow(df.corr())
plt.savefig('corr')

# show correlation with 'Class' variable
df.corrwith(df['Class'])

# Class balance:
df.Class.value_counts()
# 442/(15162+442) = 2.8%

# Next:
# over or under-sample the data
# plug into logistic regression
# plug into decision tree
# plug into random forest
# plug into gradient boosted random forest
# Finished on Friday at 3pm. 1.5 hours on EDA down. 
# Take-away: have EDA snippets available. Or last EDA.

# What follows is from source [1]:
#
# Set X and y
cols = list(columns.copy())
cols.remove('Class')
cols.remove('ID')
X = df.loc[:, cols]
y = df.loc[:, df.columns == 'Class']


# Oversample
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

# Logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(os_data_X, os_data_y)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# What follows is from source [2]:
# The estimated coefficients:
print(logreg.coef_)

# The estimated coefficients after normalizing:
# (note that dividing X by its standard deviation will require multiplying
# the weights by the standard deviation)
print((np.std(X_test, 0)*logreg.coef_[0]).sort_values(ascending=False))

# So the top 4 indicators of wellness are Uniformity of Cell Size, Marginal 
# Adhesion, Clump Thickness, and Mitoses, and then Single Epithelial Cell Size is 
# the top indicator of illness. 

# The following is from source [3]
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(os_data_X, np.ravel(os_data_y));

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)# Calculate the absolute errors
errors = abs(predictions - np.ravel(y_test))# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#Mean Absolute Error: 0.0 degrees.

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / np.ravel(y_test))# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Pulling from source [2]
# produce binary predictions based on threshold, check confusion matrix
threshold = .4 # .3 and .4 have the lowest errors
predictions_thresh = (predictions >= threshold)
confusion_matrix_rf = confusion_matrix(y_test, predictions_thresh)
print(confusion_matrix_rf)

type(predictions)
type(y_pred)
len(predictions)
len(y_pred)
predictions.shape
y_pred.shape

?logreg.predict_proba
logreg.predict_proba(X_test)[:,1]
logreg.predict_proba(X_test)#[:,1]

# ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
rf_roc_auc = roc_auc_score(y_test, predictions_thresh)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, predictions)
plt.figure()
plt.plot(rf_fpr, rf_tpr, label='Logistic Regression (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('RF_ROC')
plt.show()

# Back to source [3]
# Saving feature names for later use
feature_list = list(X_test.columns)
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(os_data_X, np.ravel(os_data_y))
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


########################
# Dear god, I used a regression forest instead of classification!
#######################
# from source [4]:
#
from sklearn.ensemble import RandomForestClassifier
# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(os_data_X, os_data_y)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
preds = clf.predict(X_test)

# View the predicted probabilities of the first 10 observations
preds_proba = clf.predict_proba(X_test)

# Create actual english names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

# Create confusion matrix
#pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
confusion_matrix_3 = confusion_matrix(y_test, preds)
print(confusion_matrix_3)



# source [1]: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# source[2]: https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model
# source[3]: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# source[4]: https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
