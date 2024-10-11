

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

list = pd.read_csv('D:\\CS 484 (Intro To ML)\\Assignment 1\\FRAUD.csv')

empirical_fraud_rate = round(list['FRAUD'].mean(), 4)
print("Empirical Fraud Rate:", empirical_fraud_rate)
seed = 202303484
complete_data = list.dropna()
train_proportion = 0.8
num_train_samples = int(len(complete_data) * train_proportion)
train_data = complete_data.sample(n=num_train_samples, random_state=seed)
test_data = complete_data.drop(train_data.index)


train_count = len(train_data)
test_count = len(test_data)
print("\nNumber of Observations in Training Partition:", train_count)
print("Number of Observations in Testing Partition:", test_count)
features = ['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']
target = 'FRAUD'
misclassification_rates_train = []
misclassification_rates_test = []

for n_neighbors in range(2, 8):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_data[features], train_data[target])
    
    train_predictions = knn.predict(train_data[features])
    test_predictions = knn.predict(test_data[features])
    
    train_misclassification_rate = 1 - accuracy_score(train_data[target], train_predictions)
    test_misclassification_rate = 1 - accuracy_score(test_data[target], test_predictions)
    
    misclassification_rates_train.append(train_misclassification_rate)
    misclassification_rates_test.append(test_misclassification_rate)

for n_neighbors, train_misclassification, test_misclassification in zip(range(2, 8), misclassification_rates_train, misclassification_rates_test):
    print(f"\nNeighbors: {n_neighbors}, Train Misclassification Rate: {train_misclassification:.4f}, Test Misclassification Rate: {test_misclassification:.4f}")


best_neighbors = min(range(2, 8), key=lambda i: misclassification_rates_test[i-2])
print("\nNumber of Neighbors with Lowest Test Misclassification Rate:", best_neighbors)


focal_observation = pd.DataFrame({
    'DOCTOR_VISITS': [8],
    'MEMBER_DURATION': [178],
    'NUM_CLAIMS': [0],
    'NUM_MEMBERS': [2],
    'OPTOM_PRESC': [1],
    'TOTAL_SPEND': [16300]
})

knn_selected = KNeighborsClassifier(n_neighbors=best_neighbors)
knn_selected.fit(train_data[features], train_data[target])

neighbors_indices = knn_selected.kneighbors(focal_observation, return_distance=False)
neighbors = train_data.iloc[neighbors_indices[0]]

fraud_probability = knn_selected.predict_proba(focal_observation)[0][1]

print("\nNeighbors' Observation Values:")
print(neighbors)
print("Predicted Probability of Fraud:", fraud_probability)
