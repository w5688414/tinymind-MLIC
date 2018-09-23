from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from measure import *
from data_helper import *
from model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


train_path = 'visual_china_train1.csv'
train_df = pd.read_csv(train_path)
for i in range(35000):
    train_df['img_path'].iloc[i] = train_df['img_path'].iloc[i].split('/')[-1]

img_paths = list(train_df['img_path'])

y_train = load_ytrain('tag_train.npz')

nub_train = 35000  #可修改，前期尝试少量数据验证模型
X_train = np.zeros((nub_train,224,224,3),dtype=np.uint8)
i = 0

for img_path in img_paths[:nub_train]:
    img = Image.open('train/' + img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224,224))
    arr = np.asarray(img)
    X_train[i,:,:,:] = arr
    i += 1


batch_size = 16
epochs = 100


X_train2,X_val,y_train2,y_val = train_test_split(X_train, y_train[:nub_train], test_size=0.2, random_state=2018)

train_datagen = ImageDataGenerator(width_shift_range = 0.1, 
                                 height_shift_range = 0.1, 
                                 zoom_range = 0.1)
val_datagen = ImageDataGenerator()     #验证集不做图片增强

train_generator = train_datagen.flow(X_train2,y_train2,batch_size=batch_size,shuffle=False) 
val_generator = val_datagen.flow(X_val,y_val,batch_size=batch_size,shuffle=False)

checkpointer = ModelCheckpoint(filepath='weights_best_simple_model.hdf5', 
                            monitor='val_fmeasure',verbose=1, save_best_only=True, mode='max')
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=2,verbose=1,mode='max')
model = build_model(6941)
model.summary()
model.compile(optimizer = 'adam',
           loss='binary_crossentropy',
           metrics=['accuracy',fmeasure,recall,precision])



history = model.fit_generator(generator=train_generator,steps_per_epoch=X_train2.shape[0]//batch_size,
       validation_data = val_generator,validation_steps=X_val.shape[0]//batch_size,
       epochs=epochs,
       callbacks=[checkpointer,reduce],
       verbose=1)