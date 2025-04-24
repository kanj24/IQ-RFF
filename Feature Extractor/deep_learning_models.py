import numpy as np
import tensorflow as tf


from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Flatten, Layer, Embedding, Activation
from tensorflow.keras.layers import LayerNormalization, Dropout, MultiHeadAttention, Add, GlobalAveragePooling1D, AveragePooling1D,Concatenate,LSTM,Conv2D,ReLU,AveragePooling2D,Conv1D

from dataset_preparation import ChannelIndSpectrogram



def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss_fcn(margin):
    def loss_fn(y_true, y_pred):
        # Assume each embedding is of length D
        d = tf.shape(y_pred)[1] // 3
        anchor = y_pred[:, :d]
        positive = y_pred[:, d:2*d]
        negative = y_pred[:, 2*d:]

        # Compute squared distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Compute triplet loss
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        return tf.reduce_mean(loss)
    return loss_fn

def resblock_spec(x, kernelsize, filters, first_layer = False):

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

def resblock_slice(x, kernelsize, filters, first_layer = False):

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

def TransformerEncoderBlock(x, d_model, ff_dim, num_heads, dropout):
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

    
class TripletNet():
    def __init__(self):
        pass

    def triplet_loss(self,x,margin):
        anchor,positive,negative = x

        pos_dist = K.sum(K.square(anchor-positive),axis=1)
        neg_dist = K.sum(K.square(anchor-negative),axis=1)

        basic_loss = pos_dist-neg_dist + margin
        loss = K.maximum(basic_loss,0.0)
        return loss  
        
    def create_triplet_net(self,feature_extractor, input_shape, margin):

        #########For Resnet spectrogram only##########
        # input_1 = Input(shape=(input_shape[1],input_shape[2],input_shape[3]))
        # input_2 = Input(shape=(input_shape[1],input_shape[2],input_shape[3]))
        # input_3 = Input(shape=(input_shape[1],input_shape[2],input_shape[3]))
        ##################################



        ###########For resenet slice, txfr, and lstm ####################
        input_1 = Input(shape=(input_shape[1],input_shape[2]))
        input_2 = Input(shape=(input_shape[1],input_shape[2]))
        input_3 = Input(shape=(input_shape[1],input_shape[2]))
        ###############################################



        
        A = feature_extractor(input_1)
        P = feature_extractor(input_2)
        N = feature_extractor(input_3)        

   
        #########For Resnet only##########
        #loss = Lambda(lambda x: self.triplet_loss(x, margin=margin))([A, P, N])
        #return Model(inputs=[A, P, N], outputs=loss)
        ##################################



        ###########For txfr and lstm ####################
        output = Concatenate(axis=1)([A, P, N])
        return Model(inputs=[input_1, input_2, input_3], outputs=output)
        #################################################


    
    def feature_extractor_resnet_spec(self, datashape):
            
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        
        x = resblock_spec(x, 3, 32)
        x = resblock_spec(x, 3, 32)

        x = resblock_spec(x, 3, 64, first_layer = True)
        x = resblock_spec(x, 3, 64)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
  
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model   
    
    def feature_extractor_resnet_slice(self, datashape):
            
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2]]))

        x = Conv1D(32, kernel_size=7, strides=2, activation='relu', padding='same')(inputs)
    
        # Residual blocks
        x = resblock_slice(x, kernelsize=3, filters=32)
        x = resblock_slice(x, kernelsize=3, filters=32)
        x = resblock_slice(x, kernelsize=3, filters=64, first_layer=True) 
        x = resblock_slice(x, kernelsize=3, filters=64)

        # Pooling
        x = AveragePooling1D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
  
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

     
    
    def feature_extractor_transformer(self,input_shape,num_heads, ff_dim, num_layers, output_dim,d_model,dropout):
        
        inputs = Input(shape=(input_shape[1], input_shape[2]))
        x = Dense(d_model)(inputs)
        x = LayerNormalization()(x)
        x = AveragePooling1D(pool_size=20)(x) #for slices only
        
        for _ in range(num_layers):
            x = TransformerEncoderBlock(x,d_model=d_model, ff_dim=ff_dim, num_heads=num_heads, dropout=dropout)


        x = GlobalAveragePooling1D()(x)
        x = Dense(d_model,activation = 'sigmoid')(x) #remove activation if necessary
        x = Dense(output_dim, activation=None)(x)
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)


        model = Model(inputs=inputs, outputs=outputs)

        return model
    
    def feature_extractor_lstm(self,input_shape,neurons, plusoneLSTM, output_dim,dropout):
        
        inputs = Input(shape=(input_shape[1], input_shape[2]))
        x = LayerNormalization()(inputs)
        x = x[:, :int(input_shape[1]/4), :] #only for slices
        
        x = LSTM(neurons, return_sequences=True, recurrent_dropout=dropout)(x)
        #x = LayerNormalization()(x)
        
        for _ in range(plusoneLSTM):
            x = LSTM(neurons, return_sequences=True, recurrent_dropout=dropout)(x)
            #x = LayerNormalization()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(output_dim, activation=None)(x)
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model
        
    
    def get_triplet(self):
        n = a = self.dev_range[np.random.randint(len(self.dev_range))]
        
        while n == a:
            n = self.dev_range[np.random.randint(len(self.dev_range))]
        a, p = self.call_sample(a), self.call_sample(a)
        n = self.call_sample(n)
        
        
        return a, p, n

          
    def call_sample(self,label_name):

        num_sample = len(self.label)
        idx = np.random.randint(num_sample)
        while self.label[idx] != label_name:
            idx = np.random.randint(num_sample) 
        return self.data[idx]


    def create_generator(self, batchsize, dev_range, data, label):
        self.data = data
        self.label = label
        self.dev_range = dev_range

        
        while True:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batchsize):
                a, p, n = self.get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')

            label = np.ones(batchsize)
            yield [A, P, N], label  




