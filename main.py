"""
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def lienar_regression(data):
    # vetores de valores x e y
    ef = [data["ejection_fraction"].values]
    sc = [data["serum_creatinine"].values]

    ejection_fraction = np.array(ef).reshape(-1,1) #x
    serum_creatinine = np.array(sc).reshape(-1,1) #y

    # criar modelo linear e otimizar
    model = LinearRegression().fit(ejection_fraction, serum_creatinine)

    # extrair coeficientes
    slope = model.coef_
    intercept = model.intercept_

    # definir as variáveis
    x = ejection_fraction
    y1 = serum_creatinine
    y2 = slope * ejection_fraction + intercept

    # plotar
    plt.title('Linear Regression')
    plt.ylabel('Serum Creatinine')
    plt.xlabel('Ejection Fraction')
    plt.plot(x, y2, 'r')
    plt.plot(x, y1, '.k')
    plt.show()


death_data = pd.read_csv('data/death_dataset.csv')
death_data.head()

all_data = pd.read_csv('data/heart_failure_dataset.csv')
all_data.head()

alive_data = pd.read_csv('data/alive_detaset.csv')
alive_data.head()

lienar_regression(death_data)
lienar_regression(alive_data)
lienar_regression(all_data)
