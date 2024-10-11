import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, mean_squared_error
from itertools import combinations

file_path = 'Homeowner_Claim_History.xlsx'
data = pd.read_excel(file_path, sheet_name='HOCLAIMDATA')

data['Frequency'] = data['num_claims'] / data['exposure']

missfreq = data['Frequency'].isnull().sum()
print("missing_frequency",missfreq)

uniq_freq = data['Frequency'].unique()


no_uniq_freq = len(uniq_freq)


sortuniqfreq = np.sort(uniq_freq)

nozerofreq = len(data[data['Frequency']==0])


bins = [-float('inf'),0, 1, 2, 3, float('inf')]


data['Frequency_Group'] = pd.cut(data['Frequency'], bins, labels=[0,1,2,3,4])



clean_data = data.dropna(subset=['Frequency_Group'])


data_train = data[data['policy'].str.startswith(('A', 'G', 'P'))].reset_index(drop=True)
data_test = data[~data['policy'].str.startswith(('A', 'G', 'P'))].reset_index(drop=True)

count_train_group = data_train['Frequency_Group'].value_counts().sort_index()
count_test_group = data_test['Frequency_Group'].value_counts().sort_index()

print('Training Data :',count_train_group)
print( 'Testing Data : ',count_test_group)
print( 'Missing Frequency : ', missfreq)

categorical_predictors = [
    'f_aoi_tier', 'f_fire_alarm_type', 'f_marital',
    'f_mile_fire_station', 'f_primary_age_tier',
    'f_primary_gender', 'f_residence_location'
]

substs = []
for r in range(1, len(categorical_predictors) + 1):
    substs.extend(combinations(categorical_predictors, r))

total_subsets = len(substs)
print(f"Total subsets: {total_subsets}")

result_df = pd.DataFrame(columns=["Subset", "AIC", "BIC", "Accuracy", "Rase"])

for s in substs:  
    subset_list = list(s)  
    encoder = OneHotEncoder(drop='first')
    encoder.fit(data_train[subset_list])

    
    X_train_encoded = encoder.transform(data_train[subset_list])
    X_test_encoded = encoder.transform(data_test[subset_list])

    
    X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out())
    X_train_encoded_df['Intercept'] = 1

    X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out())
    X_test_encoded_df['Intercept'] = 1

    
    y_train = data_train['Frequency_Group']
    y_test = data_test['Frequency_Group']

    
    logit_model = sm.MNLogit(y_train, X_train_encoded_df)
    logit_result = logit_model.fit()

    
    aic_value = logit_result.aic
    bic_value = logit_result.bic

    
    y_pred_prob = logit_result.predict(X_test_encoded_df)
    y_pred = y_pred_prob.values.argmax(axis=1)  

    accuracy = accuracy_score(y_test, y_pred)
    rase = np.sqrt(mean_squared_error(y_test, y_pred_prob.idxmax(axis=1)))

    
    result = pd.DataFrame({"Subset": [subset_list], "AIC": [aic_value], "BIC": [bic_value], "Accuracy": [accuracy], "Rase": [rase]})
    result_df = pd.concat([result_df, result], ignore_index=True)

print(result_df)



min_aic_row = result_df.loc[result_df['AIC'].idxmin()]


lowest_aic_value = min_aic_row['AIC']
model_with_lowest_aic = min_aic_row['Subset']

print("The lowest AIC value is:", lowest_aic_value)
print("The model that produces this AIC value uses the following subset of predictors:\n", model_with_lowest_aic)



min_bic_row = result_df.loc[result_df['BIC'].idxmin()]


lowest_bic_value = min_bic_row['BIC']
model_with_lowest_bic = min_bic_row['Subset']

print("The lowest BIC value is:", lowest_bic_value)
print("The model that produces this BIC value uses the following subset of predictors:\n", model_with_lowest_bic)



max_accuracy_row = result_df.loc[result_df['Accuracy'].idxmax()]


highest_accuracy_value = max_accuracy_row['Accuracy']
model_with_highest_accuracy = max_accuracy_row['Subset']

print("The highest accuracy value on the testing partition is:", highest_accuracy_value)
print("The model that produces this accuracy value uses the following subset of predictors:\n", model_with_highest_accuracy)



min_rase_row = result_df.loc[result_df['Rase'].idxmin()]


lowest_rase_value = min_rase_row['Rase']
model_with_lowest_rase = min_rase_row['Subset']

print("The lowest RASE value on the testing partition is:", lowest_rase_value)
print("The model that produces this RASE value uses the following subset of predictors:\n", model_with_lowest_rase)



sorted_by_aic = result_df.sort_values('AIC')
sorted_by_bic = result_df.sort_values('BIC')


optimal_subset_aic = sorted_by_aic.iloc[0]
optimal_subset_bic = sorted_by_bic.iloc[0]


print("Optimal subset by AIC:", optimal_subset_aic)
print("Optimal subset by BIC:", optimal_subset_bic)



