import keras
from keras.layers import Conv2D, Dense, Flatten, ReLU, MaxPool2D, concatenate

class DeepQNetwork(keras.Model):

    def __init__(self, input_dims, n_actions,input_conv_dims=None, isV1 = True):
        super(DeepQNetwork, self).__init__()
        self.isV1 = isV1
        if self.isV1:
            self.fc1 = Dense(16, activation='relu', input_shape=(None, input_dims))
            self.fc2 = Dense(16, activation='relu')
            self.action_layer = Dense(n_actions, activation=None)
            return

        input_conv_shape = (None, input_conv_dims[0], input_conv_dims[1], input_conv_dims[2])
        self.conv1 = Conv2D(8, (4, 4), strides=2, activation='relu', padding='same', input_shape=input_conv_shape)
        self.conv2 = Conv2D(8, (4, 4), strides=2, activation='relu', padding='same')
        self.flat = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(16, activation='relu', input_shape=(None, input_dims))
        self.fc4 = Dense(16, activation='relu')
        self.action_layer = Dense(n_actions, activation=None)

    def call(self, statesInput):
        if self.isV1:
            x = self.fc1(statesInput)
            x = self.fc2(x)
            x = self.action_layer(x)
            return x
        else:
            conv_state, state = statesInput
            #conv block
            x1 = self.conv1(conv_state)
            x1 = self.conv2(x1)
            x1 = self.flat(x1)
            x1 = self.fc1(x1)
            x1 = self.fc2(x1)
            
            #features block
            x2 = self.fc3(state)
            
            #final block
            x = concatenate([x1, x2])
            x = self.fc4(x)
            x = self.action_layer(x)
            
            return x