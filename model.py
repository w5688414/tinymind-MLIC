from keras.layers import *
from keras.models import *

def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def build_model(out_dims, input_shape=(224, 224, 3)):
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