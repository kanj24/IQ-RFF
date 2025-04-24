import tensorflow as tf

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, ReLU, Add, AveragePooling1D, Flatten, LayerNormalization, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling1D, MultiHeadAttention, Add, GlobalAveragePooling1D,Dense,Flatten,AveragePooling1D, LSTM, AveragePooling2D

def resblock(x, kernelsize, filters, first_layer = False):

    if first_layer:
        fx = Conv1D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv1D(filters, kernelsize, padding='same')(fx)
        
        x = Conv1D(filters, 1, padding='same')(x)
        
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv1D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv1D(filters, kernelsize, padding='same')(fx)
        
        
        out = Add()([x,fx])
        out = ReLU()(out)

    return out 


def resnet_model(input_shape, num_classes):
            
        inputs = Input(shape=(input_shape[1], input_shape[2])) 
    
        # Initial convolution
        x = Conv1D(32, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
    
        # Residual blocks
        x = resblock(x, kernelsize=3, filters=32)
        x = resblock(x, kernelsize=3, filters=32)
        x = resblock(x, kernelsize=3, filters=64, first_layer=True)  # Downsample
        x = resblock(x, kernelsize=3, filters=64)

        # Pooling
        x = AveragePooling1D(pool_size=2)(x)
        
        x = Flatten()(x)

        outputs = Dense(num_classes, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             

def TransformerEncoderBlock(x, d_model, ff_dim, num_heads, dropout):
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Feedforward
    ffn_output = Dense(ff_dim, activation='relu')(x)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization()(x)
    return x

def transformer_model(input_shape, num_heads, ff_dim, num_layers, num_classes,d_model):
 
    inputs = Input(shape=(input_shape[1],input_shape[2]))
    #x = GaussianNoise(0.2)(inputs)
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)
    x = AveragePooling1D(pool_size=20)(x) # applies only to slices
    #x = x[:, :3660, :] 
        
    for _ in range(num_layers):
        x = TransformerEncoderBlock(x,d_model=d_model, ff_dim=ff_dim, num_heads=num_heads, dropout=0.0)
        
    
    
    x = GlobalAveragePooling1D()(x) 
    x = Dense(d_model,activation='tanh')(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def lstm_model(input_shape,neurons, addlayers, num_classes,dropout):
    
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = LayerNormalization()(inputs)
    #x = GaussianNoise(0.3)(inputs)
    x =  x[:, :int(input_shape[1]/4), :] # applies only to slices
            

    for _ in range(addlayers):
        x = LSTM(neurons, return_sequences=True,recurrent_dropout=dropout)(x)
        x = LayerNormalization()(x)

    x = LSTM(neurons, return_sequences=True,recurrent_dropout=dropout)(x)
    x = LayerNormalization()(x)

        
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def resblockspec(x, kernelsize, filters, first_layer = False):

    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        x = Conv2D(filters, 1, padding='same')(x)
        
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        
        out = Add()([x,fx])
        out = ReLU()(out)

    return out 

def resnet_model_spec(input_shape, num_classes):
            
        inputs = Input(shape=([input_shape[1],input_shape[2],input_shape[3]]))
        
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        
        x = resblockspec(x, 3, 32)
        x = resblockspec(x, 3, 32)

        x = resblockspec(x, 3, 64, first_layer = True)
        x = resblockspec(x, 3, 64)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)

        outputs = Dense(num_classes, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

