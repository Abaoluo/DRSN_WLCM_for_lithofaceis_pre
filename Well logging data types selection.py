# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:49:40 2022

@author: abaoluo
"""
import pandas
import numpy
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set_style("whitegrid")

df_tr = pandas.read_excel("essay_input.xlsx", header = None)
names = ['RD', 'GR', 'CAL', 'RS', 'RSFL', 'AC', 'CNC', 'DEN']

X = numpy.float_(df_tr.iloc[1:, 1:9])
std = MinMaxScaler()
X = std.fit_transform(X)
Y = numpy.int_(df_tr.iloc[1:, 9])

chi_values, p_values = chi2(X, Y)
df = pandas.DataFrame({"Feature": names, "Chi-Squared": chi_values})
df = df.sort_values(by="Chi-Squared", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(df["Feature"], df["Chi-Squared"], palette =sns.color_palette("Set2"))
#sns.histplot(data=df, bins=20, color="green", alpha=0.8)
sns.despine()

plt.xlabel("Feature")
plt.ylabel("Chi-Squared")
plt.title("Chi-Square Test on Well Logging Data")

plt.show()
scores = pandas.DataFrame(X_new.scores_, names[:])

