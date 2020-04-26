# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:30:08 2020

@author: mpoujol
"""


#%%
'''
COMPLETE THIS PART BEFORE STARTING
'''

# In[1]:


import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

# In[2]:


from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K
from keras.utils.np_utils import to_categorical

#%%  
# Usefull variables
dataFolder = "P:\mpoujol\modeles\Python\Projets\EUvsVirus\Data"    #data path
modelFolder = "P:\mpoujol\modeles\Python\Projets\EUvsVirus\Models"
dirSep="\\"    #separator
fileType = '*.txt'

varNames = ["sinPitch","cosPitch","sinRoll","cosRoll","sinYaw","cosYaw"]
nbVar = len(varNames)
datasetFilename = "DataSet3_shift_12_12.npz"


#%%  
# Deep Learning hyper-parameters : several values have been tried for each parameter
# Here the test size is small because we have very few input data and because among the different techniques used to prevent 
# (or restrict) overfitting (early-stopping, regularization), there was one doing a sort of cross-validation
test_size=0.2                   # percentage of input data used for testing 
nbEpochs = 100                  # maximum number of epochs (generaly training is stopped much before)
batch_size=64                   # size of batches used for training with stochastic gradient descent
patience= 5                     # patience for early stopping 
lrnRate = 2e-2                  # learning rate 
nbFilters1 = 50                 # nb of convolutionnal filter for the first layer
filterLgth1 = 6                 # length of convolutionnal filter for the first layer
filterLgth2 = 3                 # length of convolutionnal filter for the second layer
l1 = 0.01                       # l1 regularization parameter
l2 = 0.01                       # l2 regularization parameter
verbose = 1
random_state=42                 # random seed to make random drawing reproducible
saveNNFilnm = "modelCNN1D"+ str(nbFilters1) + '_' + datasetFilename.split(".")[0]
bestModelFilnm = 'BestCurrentModel'


# In[17]:
# Reading of scaled data (training + test data) in a npz file where they have been stored
npzFiles = np.load(dataFolder + dirSep + datasetFilename, allow_pickle=True)
Xseq = npzFiles['X']                                 # reading of input data sample 
Y = npzFiles['Y']                                # reading of target data (also used as imput since we want to forecast them)

# Sizes of input data tensor 
nbSeqTot, SEQLENGTH, nbVar = Xseq.shape

# In[17]:

# Splitting of data in train and test datasets with a random draw but repeatable
# (for comparison of trainings) since the random seed (random_state) is fixed
indices = np.arange(0, nbSeqTot)
X_train, X_test, Y_train, Y_test, idX_train, idX_test = train_test_split(Xseq, Y, indices, test_size=test_size, random_state=random_state)

iSeqTrainRisk = np.where(Y_train>0)[0] 
nbSeqTrainRisk = len(iSeqTrainRisk)
iSeqTestRisk = np.where(Y_test>0)[0] 
nbSeqTestRisk = len(iSeqTestRisk)

# In[18]:
# Building of a light weight CNN with 1D convolutionnal layers 
model = Sequential()
model.add(Conv1D(filters= nbFilters1, 
                 kernel_size= filterLgth1,
                 padding = 'same',
                 input_shape=(SEQLENGTH, nbVar),
                 activation='relu'))
model.add(Conv1D(filters= nbFilters1, kernel_size= filterLgth1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= nbFilters1, kernel_size= filterLgth2, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))     # final logistic-like binary classification

model.summary()

# In[28]:

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# In[30]:
callbacks_list = [
    ModelCheckpoint(
        filepath=modelFolder+dirSep+bestModelFilnm+".h5",
        monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=patience)
]


history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nbEpochs,
          callbacks=callbacks_list,
          validation_data=(X_test, Y_test),
          verbose=1)

# Reload best model
model = load_model(modelFolder + dirSep + bestModelFilnm + ".h5") 
print("Best model reloaded from disk")
model.save(modelFolder + dirSep + saveNNFilnm + ".h5")
print("Best model saved on disk with more explicit name")

# Predictions on test data
Y_Pred = np.round(model.predict(Xseq))


display = 1
if display:
    plt.figure(figsize=(10,2))
    plt.xlabel("Safe and Risky situations")
    plt.plot(Y,'b',linewidth = 0.5)
    plt.plot(Y_Pred,'or', markersize=0.4)
    plt.legend(['Ground Truth','Prediction'])
    plt.title('Predictions vs. ground truth of Safe and Risky situations',fontweight="bold")
    plt.xlim(0,10* np.ceil(nbSeqTot/10))
    plt.tight_layout()
    ax=plt.gca()
    ax.axes.yaxis.set_ticklabels([])

cf_matrix = confusion_matrix(Y, Y_Pred) 
  
print('Confusion Matrix :')
print(cf_matrix) 
print('Accuracy Score :',accuracy_score(Y, Y_Pred)) 
print('Report : ')
print(classification_report(Y, Y_Pred)) 

# Plotting of confusion matrix
import seaborn as sns
plt.figure()
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')