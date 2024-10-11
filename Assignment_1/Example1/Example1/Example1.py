
import pandas as pd


filepath = 'D:\\CS 484 (Intro To ML)\\Assignment 1\\Gamma4804.csv'

readfile = pd.read_csv(filepath) 

df = pd.DataFrame(readfile)

average = df['x'].mean()
count=len(df)
standard_deviation = df['x'].std()
minimum = df['x'].min()
maximum = df['x'].max()
tenth_percentile = df['x'].quantile(0.1)
twenty_fifth_percentile = df['x'].quantile(0.25)
median = df['x'].quantile(0.5)
seventy_fifth_percentile = df['x'].quantile(0.75)
ninetyth_percentile = df['x'].quantile(0.9)

print('Mean = ',round(average,7))
print('Count = ',count)
print('Standard Deviation = ', round(standard_deviation,7))
print('Minimum = ', round(minimum,7))
print('Maximum = ', round(maximum,7))
print('10th Percentile = ',round(tenth_percentile,7))
print('25th Percentile = ',round(twenty_fifth_percentile,7))
print('Median = ',round(median,7))
print('75th Percentile = ',round(seventy_fifth_percentile,7))
print('90th Percentile = ',round(ninetyth_percentile,7))


