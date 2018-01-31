import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams
from pylab import savefig

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

hr2 = hr.reindex(columns=[age, gender, pay, source, term_why, status, score])
print (age)
print (pay)

#agemiss_flag = 1 if age=nan

age_nomiss = age.dropna()
print(age_nomiss.head(10))

pay_nomiss = pay.dropna()
print(pay_nomiss.head(10))

print (age_nomiss)
print (pay_nomiss)

rcParams['figure.figsize'] = 5, 4 #this is the size of the plot
sb.set_style('whitegrid') #this is the style: white grid

hr2.plot(kind='scatter', x=age, y=pay, c=['darkgray'], s=150)
#NotImplemented Error: Index._join_level on non-unique index is not implemented
plt.xlabel('Age')
plt.ylabel('Pay')
plt.title('Pay rate by Age')
plt.show()
