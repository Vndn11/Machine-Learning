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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


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

y_pred_train_clf = clf.predict(X_train)
y_pred_test_clf = clf.predict(X_test)

accuracy_train_clf = accuracy_score(y_train, y_pred_train_clf)
report_train_clf = classification_report(y_train, y_pred_train_clf)

accuracy_test_clf = accuracy_score(y_test, y_pred_test_clf)
report_test_clf = classification_report(y_test, y_pred_test_clf)

# print(f"Accuracy of Classification Model on Training Dataset: {accuracy_train_clf}")
# print(f"Classification Report of Classification model on Test Data:\n{report_train_clf}")

# print(f"Accuracy of Classification Model on Testing Dataset: {accuracy_test_clf}")
# print(f"Classification Report of Classification model on Testing Data:\n{report_test_clf}")

# Binary logistic regression model---------------------------

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred_train_Logistic = logistic_model.predict(X_train)
y_pred_test_Logistic = logistic_model.predict(X_test)

accuracy_train_Logistic = accuracy_score(y_train, y_pred_train_Logistic)
report_train_Logistic = classification_report(y_train, y_pred_train_Logistic)

accuracy_test_Logistic = accuracy_score(y_test, y_pred_test_Logistic)
report_test_Logistic = classification_report(y_test, y_pred_test_Logistic)

# print(f"Accuracy of Binary Logistic Regression on Training Data: {accuracy_train_Logistic}")
# print(f"Classification Report of Binary Logistic Regression on Training Data:\n{report_train_Logistic}")

# print(f"Accuracy of Binary Logistic Regression on Testing Data: {accuracy_test_Logistic}")
# print(f"Classification Report of Binary Logistic Regression on Testing Data:\n{report_test_Logistic}")

# Question 1------------------------------------

rase_train_clf=np.sqrt(np.mean(np.square(y_pred_train_clf - y_train)))
rase_train_Logistic=np.sqrt(np.mean(np.square(y_pred_train_Logistic - y_train)))

rase_test_clf=np.sqrt(np.mean(np.square(y_pred_test_clf - y_test)))
rase_test_Logistic=np.sqrt(np.mean(np.square(y_pred_test_Logistic - y_test)))

print('Root Average Squared Error for Training data in Classification Model: ',rase_train_clf)
print('Root Average Squared Error for Training data in Binary Logistic Regression: ',rase_train_Logistic)

print('Root Average Squared Error for Testing data in Classification Model: ',rase_test_clf)
print('Root Average Squared Error for Testing data in Binary Logistic Regression: ',rase_test_Logistic)

# Question 2------------------------------------

y_train_clf_prob = clf.predict_proba(X_train)[:,1]
y_test_clf_prob = clf.predict_proba(X_test)[:,1]

y_train_logistic_probs = logistic_model.predict_proba(X_train)[:, 1]
y_test_logistic_probs = logistic_model.predict_proba(X_test)[:, 1]

# Calculate AUC values
auc_train_clf = roc_auc_score(y_train, y_train_clf_prob)
auc_test_clf = roc_auc_score(y_test, y_test_clf_prob)

auc_train_logistics = roc_auc_score(y_train, y_train_logistic_probs)
auc_test_logistics = roc_auc_score(y_test, y_test_logistic_probs)

print('y_test_prob',y_train_logistic_probs)

print('Area Under Curve Value for Training data in Classification Model : ',auc_train_clf)
print('Area Under Curve Value for Training data in Binary Logistic Regression :',auc_train_logistics)

print('Area Under Curve Value For Testing data in Classification Model : ',auc_test_clf)
print('Area Under Curve Value For Testing data in Binary Logistic Regression : ',auc_test_logistics)



fpr_clf , tpr_clf, _ = roc_curve(y_train, y_train_clf_prob)
roc_auc_clf = auc(fpr_clf, tpr_clf)

fpr_logistic, tpr_logistic, _ = roc_curve(y_train, y_train_logistic_probs)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

# Question 3------------------------------------

plt.figure(figsize=(10, 8))
plt.plot(fpr_clf, tpr_clf, color='darkorange', lw=2, label=f'ROC curve for Classification Tree (area = {roc_auc_clf:.2f})')
plt.plot(fpr_logistic, tpr_logistic, color='blue', lw=2, label=f'ROC curve for Logistic Regression (area = {roc_auc_logistic:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Training Data')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Question 4------------------------------------------
precision_clf , recall_clf , _ = precision_recall_curve(y_train, y_train_clf_prob)
precision_Logiatic , recall_Logistic , _ = precision_recall_curve(y_train, y_train_logistic_probs)

no_skill = len(y_train[y_train == 1]) / len(y_train)

plt.figure(figsize=(10, 6))
plt.plot(recall_clf, precision_clf, marker='.', label='Classification Tree')
plt.plot(recall_Logistic, precision_Logiatic, marker='.', label='Binary Logistic Regression')
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()


# Question 5------------------------------------------
thresholds = np.linspace(0, 1, 100)
best_threshold_clf = 0
best_f1_clf = 0
for threshold in thresholds:
    y_prediction = (y_train_clf_prob >= threshold).astype(int)
    f1 = f1_score(y_train, y_prediction)
    if f1 > best_f1_clf:
        best_f1_clf = f1
        best_threshold_clf = threshold

best_threshold_Logistic = 0
best_f1_Logistic = 0
for threshold in thresholds:
    y_prediction = (y_train_logistic_probs >= threshold).astype(int)
    f1 = f1_score(y_train, y_prediction)
    if f1 > best_f1_Logistic:
        best_f1_Logistic = f1
        best_threshold_Logistic = threshold        

print('Best Threshold for Classification tree is : ', best_threshold_clf)
print('Best F1 Score for Classification tree is : ', best_f1_clf)

print('Best Threshold for Binary Logistic Regression is : ', best_threshold_Logistic)
print('Best F1 Score for Binary Logistic Regression is : ', best_f1_Logistic)


# Question 6------------------------------------------
y_prediction_clf = (y_test_clf_prob >= best_threshold_clf).astype(int)
tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(y_test, y_prediction_clf).ravel()
misclassification_rate_clf = (fp_clf + fn_clf) / (tp_clf + tn_clf + fp_clf + fn_clf)


y_prediction_Logistic = (y_test_logistic_probs >= best_threshold_Logistic).astype(int)
tn_Logistic, fp_Logistic, fn_Logistic, tp_Logistic = confusion_matrix(y_test, y_prediction_Logistic).ravel()
misclassification_rate_Logistic = (fp_Logistic + fn_Logistic) / (tp_Logistic + tn_Logistic + fp_Logistic + fn_Logistic)


print('Misclassification Rates of Classification tree based on F1 Score: ', misclassification_rate_clf)
print('Misclassification Rates of Binary Logistic Regression based on F1 Score: ', misclassification_rate_Logistic)


# Question 7------------------------------------------
df = pd.DataFrame({'y_true': y_test, 'y_scores': y_test_clf_prob})
df = df.sort_values(by='y_scores', ascending=False)
df['cumulative_true_positives'] = df['y_true'].cumsum()
total_positives = df['y_true'].sum()
df['cumulative_gain'] = df['cumulative_true_positives'] / total_positives
df['cumulative_lift'] = df['cumulative_gain'] / (df.index + 1) * len(df) / total_positives
df['decile'] = pd.qcut(df.index, 10, labels=False) + 1
gain_table_clf = df.groupby('decile')['cumulative_gain'].last().reset_index()
lift_table_clf = df.groupby('decile')['cumulative_lift'].last().reset_index()


df = pd.DataFrame({'y_true': y_test, 'y_scores': y_test_logistic_probs})
df = df.sort_values(by='y_scores', ascending=False)
df['cumulative_true_positives'] = df['y_true'].cumsum()
total_positives = df['y_true'].sum()
df['cumulative_gain'] = df['cumulative_true_positives'] / total_positives
df['cumulative_lift'] = df['cumulative_gain'] / (df.index + 1) * len(df) / total_positives
df['decile'] = pd.qcut(df.index, 10, labels=False) + 1 
gain_table_Logistic = df.groupby('decile')['cumulative_gain'].last().reset_index()
lift_table_Logistic = df.groupby('decile')['cumulative_lift'].last().reset_index()

print('Gain Table of Classification Tree: \n',gain_table_clf)
print('\nGain Table of Binary Logistic Regression: \n', gain_table_Logistic)
print('Lift Table of Classification Tree: \n',lift_table_clf)
print('\nLift Table of Binary Logistic Regression: \n', lift_table_Logistic)
print('\nCumulative Lift of Classificaiton Tree in Decile 1: \n', lift_table_clf.loc[0]['cumulative_lift'])
print('\nCumulative Lift of Binary Logistic Regression in Decile 1: \n',lift_table_Logistic.loc[0]['cumulative_lift'])


