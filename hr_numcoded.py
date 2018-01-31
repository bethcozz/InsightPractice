import pandas as pd

address = '~/Desktop/code/hrdata.csv'
hr = pd.read_csv(address)

print(hr.head(10))
print(hr.describe())

age = hr['Age']
gender = hr['Sex']
pay = hr['Pay Rate']
source = hr['Employee Source']
term_why = hr['Reason For Term']
status = hr['Employment Status']
score = hr['Performance Score']

#First, have to define the function coded to mean replace w/ pandas
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

print 'Before Coding:'
print pd.value_counts(hr["Employment Status"])
hr["stat_coded"] = coding(hr["Employment Status"], {'Active':0,'Voluntarily Terminated':1, 'Terminated for Cause':2, 'Leave of Absence':3, 'Future Start':4, })
print '\nAfter Coding:'
print pd.value_counts(hr["stat_coded"])

