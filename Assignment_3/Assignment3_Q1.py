import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10f}'.format

def plog2p (proportion):
   if (0.0 < proportion and proportion < 1.0):
      result = proportion * numpy.log2(proportion)
   elif (proportion == 0.0 or proportion == 1.0):
      result = 0.0
   else:
      result = numpy.nan

   return (result)

def NodeEntropy (nodeCount):

   nodeTotal = numpy.sum(nodeCount)
   nodeProportion = nodeCount / nodeTotal
   nodeEntropy = - numpy.sum(nodeProportion.apply(plog2p))
   dfdd.append(nodeProportion) 
   return (nodeTotal, nodeEntropy)

def EntropyCategorySplit (target, catPredictor, splitList):

   branch_indicator = numpy.where(catPredictor.isin(splitList), 'LEFT', 'RIGHT')
   xtab = pandas.crosstab(index = branch_indicator, columns = target, margins = False, dropna = True)

   splitEntropy = 0.0
   tableTotal = 0.0

   leftStats = None
   rightStats = None

   for idx, row in xtab.iterrows():
      rowTotal, rowEntropy = NodeEntropy(row)
      tableTotal = tableTotal + rowTotal
      splitEntropy = splitEntropy + rowTotal * rowEntropy

      if (idx == 'LEFT'):
         leftStats = [rowTotal, row, rowEntropy]
      else:
         rightStats = [rowTotal, row, rowEntropy]

   splitEntropy = splitEntropy / tableTotal
  
   return(leftStats, rightStats, splitEntropy)

def takeEntropy(s):
    return s[1]



def BuildDecisionTree(data, target, nominal_predictor, ordinal_predictor, max_depth, debug='N'):
    tree = []

    def FindBestSplit(branch_data, current_depth=0):
        # Add a check for max_depth
        if current_depth >= max_depth:
            return None

        decision_tree_result = FindBestSplitRecursive(branch_data, target, nominal_predictor, ordinal_predictor, debug)
        if decision_tree_result is None:
            return None

        splitPredictor, splitEntropy, splitBranches, nodeStats = decision_tree_result
        
        left_branch_data = branch_data[branch_data[splitPredictor].isin(splitBranches[0])]
        right_branch_data = branch_data[branch_data[splitPredictor].isin(splitBranches[1])]
        
        # Recursively build left and right subtrees
        left_subtree = FindBestSplit(left_branch_data, current_depth + 1)
        right_subtree = FindBestSplit(right_branch_data, current_depth + 1)

        return {
            'splitPredictor': splitPredictor,
            'splitBranches': splitBranches,
            'nodeStats': nodeStats,
            'leftSubtree': left_subtree,
            'rightSubtree': right_subtree
        }
    
    def FindBestSplitRecursive (branch_data, target, nominal_predictor, ordinal_predictor, debug='N'):
   

        target_data = branch_data[target]

        split_summary = []

        # Look at each nominal predictor
        for pred in nominal_predictor:
            predictor_data = branch_data[pred]
            category_set = set(numpy.unique(predictor_data))
            n_category = len(category_set)

            split_list = []
            for size in range(1, ((n_category // 2) + 1)):
                comb_size = itertools.combinations(category_set, size)
                for item in list(comb_size):
                    left_branch = list(item)
                    right_branch = list(category_set.difference(item))
                    leftStats, rightStats, splitEntropy = EntropyCategorySplit (target_data, predictor_data, left_branch)
                    if (leftStats is not None and rightStats is not None):
                        split_list.append([pred, splitEntropy, left_branch, right_branch, leftStats, rightStats])

            # Determine the optimal split of the current predictor
            split_list.sort(key = takeEntropy, reverse = False)

            if (debug == 'Y'):
                print(split_list[0])

            # Update the split summary
            if split_list is not None:
                split_summary.append(split_list[0])

        # Look at each ordinal predictor
        for pred in ordinal_predictor:
            predictor_data = branch_data[pred]
            category_set = numpy.unique(predictor_data)
            n_category = len(category_set)

            split_list = []
            for size in range(1, n_category):
                left_branch = list(category_set[0:size])
                right_branch = list(category_set[size:n_category])
                leftStats, rightStats, splitEntropy = EntropyCategorySplit (target_data, predictor_data, left_branch)
                if (leftStats is not None and rightStats is not None):
                    split_list.append([pred, splitEntropy, left_branch, right_branch, leftStats, rightStats])

            # Determine the optimal split of the current predictor
            split_list.sort(key = takeEntropy, reverse = False)

            if (debug == 'Y'):
                print(split_list[0])

            # Update the split summary
            if split_list is not None:
                split_summary.append(split_list[0])

        if (debug == 'Y'):
            print(split_summary)
        # print(split_summary[0][4:6])
        # Determine the optimal predictor
        split_summary.sort(key = takeEntropy, reverse = False)
        splitPredictor = split_summary[0][0]
        splitEntropy = split_summary[0][1]
        splitBranches = split_summary[0][2:4]
        nodeStats = split_summary[0][4:6]
        print(nodeStats[0][0])
        return ([splitPredictor, splitEntropy, splitBranches, nodeStats])




    # Start building the decision tree
    root = FindBestSplit(data)
    tree.append(root)

    return tree



claim_history = pandas.read_excel('claim_history.xlsx')

n_sample = claim_history.shape[0]

print("----------------------------Part A----------------------------")
target = 'CAR_USE'
nominal_predictor = ['CAR_TYPE', 'OCCUPATION']
ordinal_predictor = ['EDUCATION']
dfdd=[]
train_data = claim_history[[target] + nominal_predictor + ordinal_predictor].dropna().reset_index(drop = True)
target_count = train_data[target].value_counts().sort_index(ascending = True)

# Recode the EDUCATION categories into numeric values
train_data['EDUCATION'] = train_data['EDUCATION'].map({'Below High School':0, 'High School':1, 'Bachelors':2, 'Masters':3, 'Doctors':4})
print(train_data['EDUCATION'].value_counts().sort_index(ascending = True))


# Part (a) Train a decision tree model with a maximum depth of 2
max_depth = 2
decision_tree = BuildDecisionTree(train_data, target, nominal_predictor, ordinal_predictor, max_depth, debug='N')
# print((decision_tree))

print("----------------------------Part B----------------------------")
# Function to predict Car Usage probabilities
def PredictCarUsage(node,fictitious_person_data):
    # print(node['splitBranches'],fictitious_person_data['OCCUPATION'])
    if node['leftSubtree'] is not None and node['rightSubtree'] is not None:
    # Node is not a leaf node, decide which branch to go
        if str(fictitious_person_data['OCCUPATION'][0]) in str(node['splitBranches'][0])  or str(fictitious_person_data['CAR_TYPE'][0]) in str(node['splitBranches'][0]) or str(fictitious_person_data['EDUCATION'][0]) in str(node['splitBranches'][0]):
            return PredictCarUsage(node['leftSubtree'],fictitious_person_data)
        else:
            return PredictCarUsage(node['rightSubtree'],fictitious_person_data)
    else:
        # print(node['nodeStats'])
    # Node is a leaf node, return the predicted probabilities
        left_prob = node['nodeStats'][0][1]['Commercial'] / node['nodeStats'][0][0]
        right_prob = node['nodeStats'][1][1]['Private'] / node['nodeStats'][1][0]
        return {'Commercial': left_prob, 'Private': right_prob}
    


# Call the function to get the Car Usage probabilities
node=decision_tree[0]
person_one_data = pandas.DataFrame({'CAR_TYPE': ['Minivan'],
                                'OCCUPATION': ['STEM'],
                                'EDUCATION': ['Masters']})
# print(person_one_data)
car_usage_probabilities = PredictCarUsage(node,person_one_data)
print("Car Usage Probabilities:")
print(f"Commercial: {car_usage_probabilities['Commercial']:.2f}")
print(f"Private: {car_usage_probabilities['Private']:.2f}")

print("----------------------------Part C----------------------------")
person_two_data = pandas.DataFrame({'CAR_TYPE': ['Pickup'],
                                'OCCUPATION': ['Student'],
                                'EDUCATION': ['High School']})
# print(person_two_data)
car_usage_probabilities = PredictCarUsage(node,person_two_data)
print("Car Usage Probabilities:")
print(f"Commercial: {car_usage_probabilities['Commercial']:.2f}")
print(f"Private: {car_usage_probabilities['Private']:.2f}")

print("----------------------------Part D----------------------------")
private_probabilities = []

# Iterate through each row in your dataset and get the predicted probability for CAR_USE = Private
for index, row in train_data.iterrows():
    person_data = pandas.DataFrame({'CAR_TYPE': [row['CAR_TYPE']],
                                    'OCCUPATION': [row['OCCUPATION']],
                                    'EDUCATION': [row['EDUCATION']]})
    car_usage_probabilities = PredictCarUsage(node, person_data)
    private_probabilities.append(car_usage_probabilities['Private'])

# Create a histogram with bin width 0.05
plt.hist(private_probabilities, bins=numpy.arange(0.0, 1.05, 0.05), edgecolor='black')# weights=histogram_weight)
plt.xlabel('Predicted Probabilities of CAR_USE = Private')
plt.ylabel('Proportion of Observations')
plt.title('Histogram of Predicted Probabilities for CAR_USE = Private')
plt.grid(True)

# Show the histogram
plt.show()


print("----------------------------Part E----------------------------")
correct_predictions = 0
total_predictions = 0

# Iterate through each row in your dataset
for index, row in claim_history.iterrows():
    person_data = pandas.DataFrame({'CAR_TYPE': [row['CAR_TYPE']],
                                    'OCCUPATION': [row['OCCUPATION']],
                                    'EDUCATION': [row['EDUCATION']]})
    
    # Get the predicted probabilities for CAR_USE
    car_usage_probabilities = PredictCarUsage(decision_tree[0], person_data)
    
    # Determine the predicted class (CAR_USE) based on probabilities
    predicted_class = 'Private' if car_usage_probabilities['Private'] >= car_usage_probabilities['Commercial'] else 'Commercial'
    
    # Get the actual class from the dataset
    actual_class = row['CAR_USE']
    
    # Check if the prediction matches the actual class
    if predicted_class == actual_class:
        correct_predictions += 1
    
    total_predictions += 1

# Calculate the misclassification rate
misclassification_rate = 1 - (correct_predictions / total_predictions)

print(f"Misclassification Rate: {misclassification_rate:.4f}")



