import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Configure printing options
np.set_printoptions(precision=10, threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.options.display.float_format = '{:,.10f}'.format

from sklearn import preprocessing, metrics, naive_bayes

# Read claim history data from an Excel file
claim_history = pd.read_excel('claim_history.xlsx')

# Select relevant columns and remove rows with missing values
train_data = claim_history[['CAR_USE', 'CAR_TYPE', 'OCCUPATION', 'EDUCATION']].dropna().reset_index(drop=True)

# Convert the target variable to a categorical type
target_variable = train_data['CAR_USE'].astype('category')

# Convert categorical features to categorical data types
car_type = train_data['CAR_TYPE'].astype('category')
occupation = train_data['OCCUPATION'].astype('category')
education = train_data['EDUCATION'].astype('category')

# Define the features for the model
features = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']

# Get the categories for each feature
feature_categories = [car_type.cat.categories, occupation.cat.categories, education.cat.categories]

# Print feature categories
print("-------------------------------------Part A-------------------------------------------------")
print('Feature Categories:')
print(feature_categories)

# Encode categorical features as ordinal values
feature_encoder = preprocessing.OrdinalEncoder(categories=feature_categories)
X_train = feature_encoder.fit_transform(train_data[features])

# Train a Categorical Naive Bayes model with alpha smoothing
naive_bayes_model = naive_bayes.CategoricalNB(alpha=0.01)
this_model = naive_bayes_model.fit(X_train, target_variable)

# Print class counts and class probabilities
print('Target Class Count:')
print(this_model.class_count_)
print('\nTarget Class Probability:')
print(np.exp(this_model.class_log_prior_))
print("-------------------------------------Part B-------------------------------------------------")

# Print empirical counts and probabilities for each feature
for i in range(len(features)):
   print('Predictor:', features[i])
   print('Empirical Counts of Features:')
   print(this_model.category_count_[i])
   print('Empirical Probability of Features given a class, P(x_i|y):')
   print(np.exp(this_model.feature_log_prob_[i]))
   print('\n')

# Predict probabilities for two fictitious persons
print("-------------------------------------Part C-------------------------------------------------")

# Fictitious person 1
person_one_data = pd.DataFrame({'CAR_TYPE': ['SUV'],
                                'OCCUPATION': ['Skilled Worker'],
                                'EDUCATION': ['Doctors']})

person_one_encoded = feature_encoder.transform(person_one_data)
person_one_pred_prob = pd.DataFrame(naive_bayes_model.predict_proba(person_one_encoded), columns='P_' + target_variable.cat.categories)

person_one_score = pd.concat([person_one_data, person_one_pred_prob], axis=1)
print('Predicted Probability for a person with Skilled Worker occupation, Doctors education, and an SUV:')
print(person_one_score)

# Fictitious person 2
person_two_data = pd.DataFrame({'CAR_TYPE': ['Sports Car'],
                                'OCCUPATION': ['Management'],
                                'EDUCATION': ['Below High School']})

person_two_encoded = feature_encoder.transform(person_two_data)
person_two_pred_prob = pd.DataFrame(naive_bayes_model.predict_proba(person_two_encoded), columns='P_' + target_variable.cat.categories)

person_two_score = pd.concat([person_two_data, person_two_pred_prob], axis=1)
print('Predicted Probability for a person with Management occupation, Below High School education, and a Sports Car:')
print(person_two_score)

# Calculate predicted probabilities on the training data
train_pred_prob = pd.DataFrame(naive_bayes_model.predict_proba(X_train), columns='P_' + target_variable.cat.categories)
train_score = pd.concat([train_data, train_pred_prob], axis=1)
print("-------------------------------------Part D-------------------------------------------------")
print("Showing Graph....")
# Display a histogram of predicted probabilities for 'Private'
pred_prob_private = train_pred_prob['P_Private']
histogram_weight = np.ones_like(pred_prob_private) / len(pred_prob_private)

plt.figure( dpi=130)
plt.hist(pred_prob_private, bins=np.arange(0.0, 1.05, 0.05), weights=histogram_weight, edgecolor='black')  
plt.xlabel('Predicted Probability of CAR_USE being Private')  
plt.ylabel('Proportion of Training Observations')  
plt.grid(alpha=0.15)
plt.show()


print("Graph showed!!!")
# Calculate misclassification rate
print("-------------------------------------Part F-------------------------------------------------")
target_pred_category = np.where(pred_prob_private >= 0.5, 'Private', 'Commercial')
misclassification_rate = 1.0 - metrics.accuracy_score(target_variable, target_pred_category)

print("Misclassification Rate of the Naïve Bayes model is:", misclassification_rate * 100)
