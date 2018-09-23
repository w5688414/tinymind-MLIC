from keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dense, Flatten,Input
)
class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Convolution2D(20, 5, 5, border_mode="same",
            input_shape=(height, width,depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

model = LeNet.build(width=img_width, height=img_height, depth=3, classes=charset_size)
model.summary()
train(model,"lenet")