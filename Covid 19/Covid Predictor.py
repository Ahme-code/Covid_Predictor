import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import xlsxwriter

#### LOAD DATA ####
data = pd.read_csv('covid.csv',sep =',')
data = data[['id','cases']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())

#### PREPARE DATA ####
print('-'*30);print('PREPARE DATA');print('-'*30)
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['cases']).reshape(-1, 1)
plt.plot(y,'-*m')
#plt.show()
polyFeat = PolynomialFeatures(degree=6)
x = polyFeat.fit_transform(x)
#print(x)

#### TRAINING DATA ####
print('-'*30);print('TRAINING DATA');print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy:{round(accuracy*100,3)} %')
y0 = model.predict(x)

#### PREDICTION ####
days = 670
print('-'*30);print('PREDICTION');print('-'*30)
print(f'Prediction - Cases after {days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[468+days]])))),)

x1 = np.array(list(range(1,468+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'-*r')
plt.plot(y0,'-*b')
plt.show()
plt.savefig('python_pretty_plot.png')
writer = pd.ExcelWriter('python_plot.xlsx', engine = 'xlsxwriter')
pd.DataFrame(y1).to_excel(writer, sheet_name='Sheet1')
worksheet = writer.sheets['Sheet1']
worksheet.insert_image('C2','python_pretty_plot.png')
writer.save()