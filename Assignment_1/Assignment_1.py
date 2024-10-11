import pandas as pd
import numpy as np
df = pd.read_csv('D:\CS 484 (Intro To ML)\Assignment 1\\Gamma4804.csv')

sum=0.0
nlist=[]
for i in df['x']:
  nlist.append(i)
  x=float(i)
  sum=sum+i
# print(nlist)  
print('Count:',len(nlist))
print('Total:',sum)  
print('Max:',max(nlist))
print('Min:',min(nlist))
print('10th Percentile:', np.percentile(nlist,10))
print('25th Percentile:', np.percentile(nlist,10))
print('Mean:',np.mean(nlist))
print('75th Percentile:', np.percentile(nlist,10))
print('Standard deviation:',np.std(nlist))


# print(nlist)
# print(count(for x in nlist if x<5 ))
# n6list=list(nlist.sort())
list6=[5,10]
for dd in list6:
  y=dd

  count=0
  ttotal=0
  total=0
  i=0
  list2=[]
  while(i<len(nlist)):
    x=nlist[i]
    yy=int(y)-int(dd)
    ttotal=ttotal+1
    if float(x) <= float(y) and float(x)>float(yy):
      count+=1
      i=i+1
    else:
      print("Range "+str(yy)+"-"+str(int(y))+": "+str(count))
      y=y+5
      total=total+count
      list2.append(count)
      count=0
      
  print("Range "+str(yy)+"-"+str(int(y))+": "+str(count))

  y=y+5
  total=total+count
  count=0
  # print(ttotal)
  print(total)
  print(list2)
    
  print('Mean',np.mean(list2))
  m1=int(np.mean(list2))
  print('Variance',np.var(list2))
  m2=int(np.var(list2))
  c=(2*m1-m2)/25
  print(str(dd)+":"+str(c))




