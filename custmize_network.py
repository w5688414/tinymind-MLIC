from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *

def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def build_model(out_dims, input_shape=(128, 128, 3)):
    inputs_dim = Input(input_shape)
    x = Lambda(lambda x: x / 255.0)(inputs_dim) #在模型里进行归一化预处理

    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
    x = bn_prelu(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = bn_prelu(x)
    x = GlobalAveragePooling2D()(x)

    dp_1 = Dropout(0.5)(x)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数

    model = Model(inputs=inputs_dim, outputs=fc2)
    return model

model = build_model(6941)
model.summary()

from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

train_datagen = ImageDataGenerator(width_shift_range = 0.1, 
                                 height_shift_range = 0.1, 
                                 zoom_range = 0.1)
val_datagen = ImageDataGenerator()     #验证集不做图片增强

batch_size = 16

train_generator = train_datagen.flow(X_train2,y_train2,batch_size=batch_size,shuffle=False) 
val_generator = val_datagen.flow(X_val,y_val,batch_size=batch_size,shuffle=False)

checkpointer = ModelCheckpoint(filepath='weights_best_simple_model.hdf5', 
                            monitor='val_fmeasure',verbose=1, save_best_only=True, mode='max')
reduce = ReduceLROnPlateau(monitor='val_fmeasure',factor=0.5,patience=2,verbose=1,mode='max')

model.compile(optimizer = 'adam',
           loss='binary_crossentropy',
           metrics=['accuracy',fmeasure,recall,precision])

epochs = 100

history = model.fit_generator(generator=train_generator,steps_per_epoch=X_train2.shape[0]//batch_size,
       validation_data = val_generator,validation_steps=X_val.shape[0]//batch_size,
       epochs=epochs,
       callbacks=[checkpointer,reduce],
       verbose=1)