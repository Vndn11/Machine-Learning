import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


test_dataset=pd.read_csv('WineQuality_Test.csv')
train_dataset=pd.read_csv('WineQuality_Train.csv')
# print('\nTesting data\n',train_dataset)
# print('\n Training data\n',test_dataset)

X_train = train_dataset[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = train_dataset['quality_grp']
X_test = test_dataset[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = test_dataset['quality_grp']

# print('X_train:',len(X_train))
print('X_test:',len(X_test))

# CLassification model---------------------------

clf= DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
accuracy_clf = accuracy_score(y_test, y_pred_clf)
print(f"Accuracy: {accuracy_clf}")

# Binary logistic regression model---------------------------

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_Logistic = logistic_model.predict(X_test)

accuracy_Logistic = accuracy_score(y_test, y_pred_Logistic)
report = classification_report(y_test, y_pred_Logistic)
print(f"Accuracy: {accuracy_Logistic}")
print(f"Classification Report:\n{report}")

# Question 1------------------------------------

rase_test_clf=np.sqrt(np.mean(np.square(y_pred_clf - y_test)))

rase_test_Logistic=np.sqrt(np.mean(np.square(y_pred_Logistic - y_test)))

# rase_train=np.sqrt(np.mean(np.square(y_train)))

print('Root Average Squared Error for test data in Classification Model: ',rase_test_clf)
print('Root Average Squared Error for test data in Binary Logistic Regression: ',rase_test_Logistic)

# Question 2------------------------------------

# y_train_clf_prob = clf.predict_proba(X_train)[:,1]
# y_test_clf_prob = clf.predict_proba(X_test)[:,1]

# y_train_logistic_probs = logistic_model.predict_proba(X_train)[:, 1]
# y_test_logistic_probs = logistic_model.predict_proba(X_test)[:, 1]

# # Calculate AUC values
# auc_train = roc_auc_score(y_train, y_train_clf_prob)
# auc_test = roc_auc_score(y_test, y_test_clf_prob)
# print('y_test_prob',y_train_logistic_probs)

# print('auc_train:',auc_train)
# print('auc_test :',auc_test)


# fpr_tree, tpr_tree, _ = roc_curve(y_train, y_train_clf_prob)
# roc_auc_tree = auc(fpr_tree, tpr_tree)

# fpr_logistic, tpr_logistic, _ = roc_curve(y_train, y_train_logistic_probs)
# roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

# # Question 3------------------------------------

# plt.figure(figsize=(10, 8))
# plt.plot(fpr_tree, tpr_tree, color='darkorange', lw=2, label=f'ROC curve for Classification Tree (area = {roc_auc_tree:.2f})')
# plt.plot(fpr_logistic, tpr_logistic, color='blue', lw=2, label=f'ROC curve for Logistic Regression (area = {roc_auc_logistic:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) - Training Data')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()


# # Confusion matrix------------------------------------
# # cm = confusion_matrix(y_test, y_pred)

# # # Display the confusion matrix
# # print("Confusion Matrix:")
# # print(cm)

# # # Plot the confusion matrix as a heatmap
# # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
# #             xticklabels=["Predicted 0", "Predicted 1"],
# #             yticklabels=["Actual 0", "Actual 1"])
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.title("Confusion Matrix")
# # plt.show()

