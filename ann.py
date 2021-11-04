import pandas as pd
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt

from pyts.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#Retrieving data from csv file
def get_data():
    dataset = r".\processed_data.csv"
    dataset = pd.read_csv(dataset, sep=",")
    return dataset

#Data normaliztion
def normalize(x):
    x = x.transpose().to_numpy()
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x

#Supervisory dataset computing
def get_supervision(data):
    q1,q2 = stat.quantiles(data, n=3) #Getting vector quantiles 
    l = []
    z = 0
    for i in data[1:]: #shifting 1 period for prediction
        if(i<q1):
            l.append([1,0,0]) #Values are one-hot-encoded
        elif(i>q2):
            l.append([0,1,0])
        else:
            l.append([0,0,1])
    l.append([0,1,0]) #2d quantile arbitrarily attributed to last supervisory vector
    return pd.DataFrame(l)

#Splitting of dataset into training (x_a) and validition (x_v) sets
#And computing of supervisory sets (y_a and y_v)
VAL_SPLIT = 38100 #As stated by Nakano et al
x = get_data()[["Return","EMA2", "EMA4", "EMA12", "EMA24", "RSI12", "RSI24", "RSI48"]]
x_a, x_v = x[:VAL_SPLIT], x[VAL_SPLIT:]
x_a, x_v = normalize(x_a), normalize(x_v)
y_a, y_v = get_supervision(x_a[0]), get_supervision(x_v[0])
x_a, x_v = x_a.transpose(), x_v.transpose()

# Building keras sequential model
WIN_X, WID_X = x_a.shape
model = Sequential()
model.add(Dense(12, input_shape=(WIN_X,WID_X), activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_a, y_a, epochs = 400, batch_size = 500, verbose = 1, validation_data=(x_v,y_v))

#Displaying model results
model.summary()
print(model.evaluate(x_v,y_v))
plt.suptitle('3-class classification, whole set normalization', y=0.98, fontsize=14)
plt.plot(history.history["accuracy"], label = "Training")
plt.plot(history.history["val_accuracy"], label = "Validation")
plt.legend( loc='lower left', borderaxespad=0.)
plt.show()  