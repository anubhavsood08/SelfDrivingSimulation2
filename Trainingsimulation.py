from utilis import *
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Step-1
path='myData'
data = importDatainfo(path)

# STEP-2 Visualization & distribution of the data 
data=balanceData(data,display=False)

# STEP-3 Preparing for processing
imagesPath,steerings=loadData(path,data)
# print(imagesPath[0],steering[0])

# STEP-4 Split of data
xTrain,xVal,yTrain,yVal=train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('Total training images',len(xTrain))
print('Total validation images',len(xVal))


# STEP -5 Augementation-Data augmentation is a technique used in 
# machine learning, particularly 
# in the context of image processing, to artificially increase 
# the size and diversity of a training dataset.
# Because data is never enough.

# STEP-6 Preprocessing


# STEP-7 Batch Generator

# STEP-8 Create the model  -Using NVIDIA Model
model= createModel()
model.summary()

# STEP-9 Training Of Model
history =model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)


# STEP-10 Save and Plot the model 
model.save('model.h5')
print('model is saved')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
# plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


