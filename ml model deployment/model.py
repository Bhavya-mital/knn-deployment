import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\DELL\Desktop\iris.csv")
print(df.head())
x=df.drop(['Species','Id'],axis=1)
y=df['Species']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=50)
from sklearn import preprocessing
x=preprocessing.StandardScaler().fit_transform(x)
from sklearn.neighbors import KNeighborsClassifier
knnmodel=KNeighborsClassifier(n_neighbors=2)
knnmodel.fit(x_train,y_train)
y_predict1=knnmodel.predict(x_test)
import pickle
pickle.dump(knnmodel,open("model.pkl","wb"))