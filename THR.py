import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequenc
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

rs = RandomState(MT19937(SeedSequence(123456789)))
tensorflow.random.set_seed(1)

train_df =pd.read_excel(
    'dataset.xlsx', sheet_name='THR',na_values='?',  usecols=[
        'Mátrix','Térhálósító','FR1','FR2','C atom arány [%]','H atom arány [%]',
        'O atom arány [%]','N atom arány [%]','P atom arány [%]','Alifás rész aránya [%]','Cikloalifás arány [%]',
        'Aromás arány [%]','LOI [V/V%]','UL94','Alkalmazott hőáram [kW/m2]', 'MLC THR [kJ/g]'])
a=train_df.head()
b=train_df
train_df = train_df.dropna()

c_atoms_ratio = train_df['C atom arány [%]']
h_atoms_ratio = train_df['H atom arány [%]']
o_atoms_ratio = train_df['O atom arány [%]']
n_atoms_ratio = train_df['N atom arány [%]']
p_atoms_ratio = train_df['P atom arány [%]']
aliphatic_ratio = train_df['Alifás rész aránya [%]'] 
cycloaliphatic_ratio = train_df['Cikloalifás arány [%]']
aromatic_ratio = train_df['Aromás arány [%]']
heatflux = train_df['Alkalmazott hőáram [kW/m2]']
LOI = train_df['LOI [V/V%]']
UL94 = train_df['UL94']
THR = train_df['MLC THR [kJ/g]']

y1 = np.array(THR)

matrixcat=train_df.Mátrix.astype("category").cat.codes
matrixcat=pd.Series(matrixcat)
hardenercat=train_df.Térhálósító.astype("category").cat.codes
hardenercat=pd.Series(hardenercat)
FR1cat=train_df.FR1.astype("category").cat.codes
FR1cat=pd.Series(FR1cat)
FR2cat=train_df.FR2.astype("category").cat.codes
FR2cat=pd.Series(FR2cat)

x1 = np.column_stack((matrixcat,hardenercat,FR1cat,FR2cat, c_atoms_ratio,h_atoms_ratio,o_atoms_ratio,n_atoms_ratio,
                      p_atoms_ratio,aliphatic_ratio,cycloaliphatic_ratio,aromatic_ratio,LOI,UL94,heatflux))
x1 = sm.add_constant(x1, prepend=True)

X_train=x1[:30]
X_val=x1[31:41]
Y_train=y1[:30]
Y_val=y1[31:41]

Y_train=np.reshape(Y_train, (-1,1))
Y_val=np.reshape(Y_val, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)
print(scaler_x.fit(X_val))
xval_scale=scaler_x.transform(X_val)
print(scaler_y.fit(Y_train))
ytrain_scale=scaler_y.transform(Y_train)
print(scaler_y.fit(Y_val))
yval_scale=scaler_y.transform(Y_val)

model = Sequential()
model.add(Dense(14, input_dim=16, kernel_initializer='normal', activation='elu'))
model.add(Dense(14, activation='elu'))
model.add(Dense(3, activation='selu'))
model.add(Dense(1, activation='linear'))
model.summary()

import tensorflow as tf
from tensorflow import keras

#print(xtrain_scale.shape)
xtrain_scale = xtrain_scale.reshape(30,16)
#print(xtrain_scale)
Xtrain = np.delete(xtrain_scale, (14), axis=1)
#print(Xtrain.shape)

#print('xval_scale:', xval_scale)
#print(xval_scale.shape)
xval_scale = xval_scale.reshape(8,16)
#print(xtrain_scale)
X = np.delete(xval_scale, (14), axis=1)
#print('X:', X)

model.compile(loss='mae',optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0008),  metrics=['mse','mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=500, batch_size=50, verbose=1, validation_split=0.2)
predictions = model.predict(xval_scale)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = scaler_y.inverse_transform(predictions)
predictions

mean_absolute_error(Y_val, predictions)

mean_squared_error(Y_val, predictions)
math.sqrt(mean_squared_error(Y_val, predictions))

Y_val
predictions

# diagram 1
a = plt.axes(aspect='equal')
plt.scatter(Y_val, predictions)
plt.xlabel('Validáció [THR (kJ/g)]')
plt.ylabel('Predikció [THR (kJ/g)]')
lims = [0, 25]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)

# diagram 2
import matplotlib.pyplot as plt

x1 = ['GER','GER APP 4%','GER RDP 4%','GER APP 2% RDP 2%','PER','PER APP 4%', 'PER RDP 4%', 'PER APP 2% RDP 2%']
y1 = predictions
y2 = Y_val

plt.scatter(x1, y1, label= "Predikció", color= "#32C2D4", 
            marker= ".", s=150)
plt.scatter(x1, y2, label= "Validáció", color= "#3259D4", 
            marker= ".", s=150)

SMALL_SIZE = 9
MEDIUM_SIZE = 12
BIGGER_SIZE = 20

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title 

plt.xticks(rotation=60)
plt.plot(x1, y1, marker='', color='#32C2D4', linewidth=0.8, linestyle='dashed', label= "Predikció",)
plt.plot(x1,y2,marker='', color='#3259D4', linewidth=0.8, linestyle='dashed', label= "Validáció",)
plt.ylabel('THR [kJ/g]')

plt.legend()
plt.show()
