import pandas as pd
from sklearn.model_selection import train_test_split

# Load your listset
list = pd.read_csv('D:\\CS 484 (Intro To ML)\\Assignment 1\\hmeq.csv')
seed = 202303484

print("*****************Part A*****************")

total_observations = len(list)
frequency_distribution_bad = list['BAD'].value_counts(dropna=False)
mean_DEBTINC = list['DEBTINC'].mean()
mean_LOAN = list['LOAN'].mean()
mean_MORTDUE = list['MORTDUE'].mean()
mean_VALUE = list['VALUE'].mean()

std_DEBTINC = list['DEBTINC'].std()
std_LOAN = list['LOAN'].std()
std_MORTDUE = list['MORTDUE'].std()
std_VALUE = list['VALUE'].std()


print("Total Observations:", total_observations)
print("Frequency:")
print(frequency_distribution_bad)
print('\nMean and Standard Deviation of Data: \n')
print('Mean of DEBTINC :', mean_DEBTINC)
print('Mean of LOAN :', mean_LOAN)
print('Mean of MORTDUE :', mean_MORTDUE)
print('Mean of VALUE :', mean_VALUE)

print('Standard Deviation of DEBTINC :', std_DEBTINC)
print('Standard Deviation of LOAN :', std_LOAN)
print('Standard Deviation of MORTDUE :', std_MORTDUE)
print('Standard Deviation of VALUE :', std_VALUE)

print("*****************Part B*****************")

train_proportion = 0.7
num_train_samples = int(len(list) * train_proportion)
train_list = list.sample(n=num_train_samples, random_state=seed) 
test_list = list.drop(train_list.index)
frequency1 = train_list['BAD'].value_counts(dropna=False)
frequency2 = test_list['BAD'].value_counts(dropna=False)

print('Training Data Observation :',len(train_list))
print("Train list:", frequency1)

print('\nTesting Data Observation :',len(test_list))
print("Testing list:", frequency2)

mean_trainlist_DEBTINC = train_list['DEBTINC'].mean()
mean_trainlist_LOAN = train_list['LOAN'].mean()
mean_trainlist_MORTDUE = train_list['MORTDUE'].mean()
mean_trainlist_VALUE = train_list['VALUE'].mean()

std_trainlist_DEBTINC = train_list['DEBTINC'].std()
std_trainlist_LOAN = train_list['LOAN'].std()
std_trainlist_MORTDUE = train_list['MORTDUE'].std()
std_trainlist_VALUE = train_list['VALUE'].std()


mean_testlist_DEBTINC = test_list['DEBTINC'].mean()
mean_testlist_LOAN = test_list['LOAN'].mean()
mean_testlist_MORTDUE = test_list['MORTDUE'].mean()
mean_testlist_VALUE = test_list['VALUE'].mean()

std_testlist_DEBTINC = test_list['DEBTINC'].std()
std_testlist_LOAN = test_list['LOAN'].std()
std_testlist_MORTDUE = test_list['MORTDUE'].std()
std_testlist_VALUE = test_list['VALUE'].std()


print("\nMean and Standard Deviation of Training partition using Simple random sampling:\n")
print('Mean of DEBTINC :', mean_trainlist_DEBTINC)
print('Mean of LOAN :', mean_trainlist_LOAN)
print('Mean of MORTDUE :', mean_trainlist_MORTDUE)
print('Mean of VALUE :', mean_trainlist_VALUE)

print('Standard Deviation of DEBTINC :', std_trainlist_DEBTINC)
print('Standard Deviation of LOAN :', std_trainlist_LOAN)
print('Standard Deviation of MORTDUE :', std_trainlist_MORTDUE)
print('Standard Deviation of VALUE :', std_trainlist_VALUE)


print("\nMean and Standard Deviation of Testing using Simple random sampling:\n")
print('Mean of DEBTINC :', mean_testlist_DEBTINC)
print('Mean of LOAN :', mean_testlist_LOAN)
print('Mean of MORTDUE :', mean_testlist_MORTDUE)
print('Mean of VALUE :', mean_testlist_VALUE)

print('Standard Deviation of DEBTINC :', std_testlist_DEBTINC)
print('Standard Deviation of LOAN :', std_testlist_LOAN)
print('Standard Deviation of MORTDUE :', std_testlist_MORTDUE)
print('Standard Deviation of VALUE :', std_testlist_VALUE)


print("*****************Part C*****************")
list['BAD'].fillna(99, inplace=True)
list['REASON'].fillna('MISSING', inplace=True)
stra1 , stra2 = train_test_split(list, test_size=0.3, random_state=seed, stratify=list[['BAD', 'REASON']])
frequency_stra1 = stra1['BAD'].value_counts()
frequency_stra2 = stra2['BAD'].value_counts()

print('\nTraining Data Observation :',len(stra1))
print("Testing list:", frequency_stra1)

print('\nTesting Data Observation :',len(stra2))
print("Train list:", frequency_stra2)


mean_trainlist_DEBTINC = stra1['DEBTINC'].mean()
mean_trainlist_LOAN = stra1['LOAN'].mean()
mean_trainlist_MORTDUE = stra1['MORTDUE'].mean()
mean_trainlist_VALUE = stra1['VALUE'].mean()

std_trainlist_DEBTINC = stra1['DEBTINC'].std()
std_trainlist_LOAN = stra1['LOAN'].std()
std_trainlist_MORTDUE = stra1['MORTDUE'].std()
std_trainlist_VALUE = stra1['VALUE'].std()


mean_testlist_DEBTINC = stra2['DEBTINC'].mean()
mean_testlist_LOAN = stra2['LOAN'].mean()
mean_testlist_MORTDUE = stra2['MORTDUE'].mean()
mean_testlist_VALUE = stra2['VALUE'].mean()

std_testlist_DEBTINC = stra2['DEBTINC'].std()
std_testlist_LOAN = stra2['LOAN'].std()
std_testlist_MORTDUE = stra2['MORTDUE'].std()
std_testlist_VALUE = stra2['VALUE'].std()


print("\nMean and Standard Deviation of Training partition using Simple random sampling:\n")
print('Mean of DEBTINC :', mean_trainlist_DEBTINC)
print('Mean of LOAN :', mean_trainlist_LOAN)
print('Mean of MORTDUE :', mean_trainlist_MORTDUE)
print('Mean of VALUE :', mean_trainlist_VALUE)

print('Standard Deviation of DEBTINC :', std_trainlist_DEBTINC)
print('Standard Deviation of LOAN :', std_trainlist_LOAN)
print('Standard Deviation of MORTDUE :', std_trainlist_MORTDUE)
print('Standard Deviation of VALUE :', std_trainlist_VALUE)


print("\nMean and Standard Deviation of Testing partition using Simple random sampling:\n")
print('Mean of DEBTINC :', mean_testlist_DEBTINC)
print('Mean of LOAN :', mean_testlist_LOAN)
print('Mean of MORTDUE :', mean_testlist_MORTDUE)
print('Mean of VALUE :', mean_testlist_VALUE)

print('Standard Deviation of DEBTINC :', std_testlist_DEBTINC)
print('Standard Deviation of LOAN :', std_testlist_LOAN)
print('Standard Deviation of MORTDUE :', std_testlist_MORTDUE)
print('Standard Deviation of VALUE :', std_testlist_VALUE)
