import matplotlib.pyplot as plt
import numpy
import pandas as pd

df = pd.read_csv('D:\\CS 484 (Intro To ML)\\Assignment 1\\Gamma4804.csv')

pd.options.display.float_format = '{:,.10f}'.format


print('------------------------------------------------ PART A -----------------------------------------------------')

Mean = df['x'].mean()
count = len(df['x'])
standard_deviation = df['x'].std()
minimum = df['x'].min()
maximum = df['x'].max()
tenth_percentile = df['x'].quantile(0.1)
twenty_fifth_percerntile = df['x'].quantile(0.25)
median = df['x'].quantile(0.5)
seventy_fifth = df['x'].quantile(0.75)
ninetith_percentile = df['x'].quantile(0.9)

print('Mean = ',round(Mean,7))
print('Count = ', count)
print('Standard Deviation = ',round(standard_deviation,7))
print('Minimum = ', minimum)
print('Maximum = ', maximum)
print('10th Percentile = ', round(tenth_percentile,7))
print('25th Percentile = ', round(twenty_fifth_percerntile,7))
print('Median = ', round(median,7))
print('75th Percentile = ', round(seventy_fifth,7))
print('90th Percentile = ', round(ninetith_percentile,7))


print('------------------------------------------------ PART B -----------------------------------------------------')


delta = [5,10,20,25]
j=0 
list_of_Shimazaki_Shinomoto_Cost_formula = []
Number_Of_Bins = []
Number_Of_Bins_with_zero_obs = []
list_of_BinBoundaries = []
bin_boundaries = []
list_of_no_of_observation = []


for i in delta :
    list1= list(df['x'])
    lower_range = 0
    upper_range = i
    count_of_observations = 0
    list2 = []
    Observations = 0
    list_of_no_of_observation_with_no_obs = []
    
    print('\n----------------- FOR DELTA : '+str(i)+'-----------------\n')
    while (len(list1)!=0) :
    #while (j<len(list1)) :
        
        if float(list1[j]) > float(lower_range) and float(list1[j]) <= float(upper_range):
            Observations = Observations + 1
            list1.pop(j)
            #j = j+1
            
            
        else : 
            #print('Range '+str(lower_range) + ' - ' + str(upper_range) + ' : ' + str(Observations))
            bin_boundaries.append(upper_range)
            lower_range = lower_range + i
            upper_range = upper_range + i
            count_of_observations = count_of_observations+Observations
            list2.append(Observations)
            Observations = 0
            #j=0
    
    count_of_observations = count_of_observations+Observations 
    list2.append(Observations)    
    # print('Range '+str(lower_range) + ' - ' + str(upper_range) + ' : ' + str(Observations))
    
    for k in list2 :
        if k !=0 :
            list_of_no_of_observation.append(k)


    print('list_of_no_of_observation',list_of_no_of_observation)
    
    Number_Of_Bins.append(len(list_of_no_of_observation))
    Number_Of_Bins_with_zero_obs.append(len(list2))
    



    df2 = pd.DataFrame(list_of_no_of_observation)

    Mean = df2[0].mean()
    
    Variance = numpy.mean(numpy.power((list_of_no_of_observation - Mean), 2))

    Shimazaki_Shinomoto_Cost_formula = (2*Mean - Variance)/(i*i)

    list_of_Shimazaki_Shinomoto_Cost_formula.append(Shimazaki_Shinomoto_Cost_formula)

    print('\nMean = ',Mean)
    print('Variance = ',Variance)
    print('Shimazaki_Shinomoto_Cost_formula = '+ str(Shimazaki_Shinomoto_Cost_formula) + '\n')
    
    list_of_BinBoundaries.append(bin_boundaries)
    bin_boundaries.clear()
print('Number_Of_Bins :',Number_Of_Bins)



print('\nMinimum Cost is : '+ str(min(list_of_Shimazaki_Shinomoto_Cost_formula)) + '\nOptimal Bin Width is : ' +str(delta[list_of_Shimazaki_Shinomoto_Cost_formula.index(min(list_of_Shimazaki_Shinomoto_Cost_formula))]))
optimal_bin_widht = delta[list_of_Shimazaki_Shinomoto_Cost_formula.index(min(list_of_Shimazaki_Shinomoto_Cost_formula))]
number_of_bins_for_optimal_bin_width_with_zero_obs = Number_Of_Bins_with_zero_obs[list_of_Shimazaki_Shinomoto_Cost_formula.index(min(list_of_Shimazaki_Shinomoto_Cost_formula))]
number_of_bins_for_optimal_bin_width = Number_Of_Bins[list_of_Shimazaki_Shinomoto_Cost_formula.index(min(list_of_Shimazaki_Shinomoto_Cost_formula))]

print('------------------------------------------------ PART C -----------------------------------------------------')


midpoint = []
listofAllObservations = list(df['x'])
print(listofAllObservations[4750:])
lower_range = 0
upper_range = optimal_bin_widht
weight = 0
total_weight = 0
density_for_midpoints = []
midpoints_with_density_zero = []

print('Total Number of Observation (N) : ' ,count_of_observations)

Nh = count_of_observations*optimal_bin_widht
print('Count_of_observations * Optimal_bin_widht = ', Nh)

print(number_of_bins_for_optimal_bin_width)

for i in range(0,number_of_bins_for_optimal_bin_width_with_zero_obs):
    midpoint.append((upper_range + lower_range)/2)
    upper_range = upper_range + optimal_bin_widht
    lower_range = lower_range + optimal_bin_widht
print(midpoint)
for i in midpoint:
    for j in listofAllObservations :
        u = (float(j) - float(i))/optimal_bin_widht
        if -0.5 < u <= 0.5:
            weight = 1
        else :
            weight = 0
        total_weight = total_weight + weight
    print('\nTotal Weight For Midpoint ' + str(i) + ' is : ' +str(total_weight) )
    density = total_weight/Nh
    # if density != 0 :
    print('Density For '+str(i) + ' is : ' + str(total_weight/Nh))
    density_for_midpoints.append((total_weight/Nh))
    # else :
    #     midpoints_with_density_zero.append(i)
    total_weight = 0


# for i in midpoints_with_density_zero:
#     midpoint.remove(i)

print('\nList of Midpoints',midpoint)
print('\nDensity List for Every Midpoint is : ',density_for_midpoints)


plt.bar(midpoint, density_for_midpoints, width=optimal_bin_widht, alpha=0.7)
plt.xlabel('Mid-Points')
plt.ylabel('Density Estimate')
plt.title('Density Estimates for Bin Width '+str(optimal_bin_widht))
plt.xticks(midpoint)
plt.show()
plt.hist(listofAllObservations, bins=number_of_bins_for_optimal_bin_width, color=None)

plt.xlabel('Value')
plt.ylabel('No of Ovbservations')
plt.title('Recommended Bin width '+str(optimal_bin_widht))
# Show grid
plt.grid(True, linestyle='--', alpha=0.6)
# Show the plot
plt.show()
