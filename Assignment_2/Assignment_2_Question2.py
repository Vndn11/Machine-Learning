import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("D:\\CS 484 (Intro To ML)\\Assignment 2\\Chinese_Bakery.csv")

print('\n################PART A################\n')
uniSetsize = len(df['Item'].unique())
MaxItems = (2 ** uniSetsize) - 1
maxAsscRules = (3 ** uniSetsize) - (2 ** (uniSetsize + 1)) + 1

print('The number of unique items in the "Item" field; ',uniSetsize)
print('The maximum number of itemsets theoretically: ',MaxItems)
print('The maximum number of association rules theoretically: ',maxAsscRules)

print('\n################PART B################\n')
basket = pd.get_dummies(df, columns=['Item'])
basket = basket.groupby(['Customer']).sum()
minSupp = 100 / len(df['Customer'].unique())
freqItemsts = apriori(basket,  min_support=minSupp, use_colnames=True)
LargeItemSize = freqItemsts['itemsets'].apply(len).max()

print('Number of Itemsets with at Least 100 Customers: ',len(freqItemsts))
print('Largest Number of Items (k) Among These Itemsets: ',str(LargeItemSize))


print('\n################PART C################\n')
rules = association_rules(freqItemsts, metric="confidence", min_threshold=0.01)
print('Number of Association Rules with at Least 1% Confidence: ',len(rules))
plt.figure(figsize=(10, 6))
cmap = plt.cm.get_cmap('coolwarm')
scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap=cmap)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence (Colored by Lift)')
plt.show()

print('\n################PART D################\n')
highConfRules = rules[rules['confidence'] >= 0.85]
highConfRules=highConfRules.assign(expected_confidence=highConfRules['support']/highConfRules['confidence'])
highConfRules = highConfRules[['antecedents', 'consequents', 'support', 'confidence', 'expected_confidence', 'lift']]
highConfRules = highConfRules.sort_values(by='lift', ascending=False)

print('Number of High-Confidence Rules: ',len(highConfRules))
print("High-Confidence Rules:")
print(highConfRules)