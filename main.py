import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# pegar os dados
heart_data = pd.read_csv('data/heart_failure_dataset.csv')
heart_data.head()

# vetores de valores x e y
ef = [heart_data["ejection_fraction"].values]
sc = [heart_data["serum_creatinine"].values]

ejection_fraction = np.array(ef).reshape(-1,1) #x
serum_creatinine = np.array(sc).reshape(-1,1) #y

# criar modelo linear e otimizar
model = LinearRegression().fit(ejection_fraction, serum_creatinine)

# extrair coeficientes
slope = model.coef_
intercept = model.intercept_

# criar a reta
x = ejection_fraction
y1 = serum_creatinine
y2 = slope * ejection_fraction + intercept

plt.title('Linear Regression') 
plt.ylabel('Serum Creatinine') 
plt.xlabel('Ejection Fraction')
plt.plot(x, y2, 'r')
plt.plot(x, y1, '.k')
plt.show()
