from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import time



from deep_learning_models import resnet_model,transformer_model, lstm_model,resnet_model_spec


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow_addons.optimizers import AdamW, CyclicalLearningRate

from tensorflow.keras.utils import to_categorical

from utils import shuffle, data_generator, load_iq, channel_ind_spectrogram, data_shape_extractor


def train(data_path, tx_range, pk_range):

    num_classes=len(tx_range)
    data, label = load_iq(data_path,tx_range,pk_range)
    label = label - 1
    label = to_categorical(label,num_classes=num_classes)
    data, label = shuffle(data,label)

    data, data_valid, label, label_valid = train_test_split(data, label, test_size=0.1, shuffle=True)

    num_train_samples = len(data)
    num_valid_samples = len(data_valid)
    batch_size = 20 # Modify as needed

    clr = CyclicalLearningRate(initial_learning_rate=1e-5,# Modify as needed
                               maximal_learning_rate=1e-3,# Modify as needed
                               step_size=200, # Modify as needed
                               scale_fn=lambda x: 1.,
                               scale_mode='cycle',
                               name='cyclic_lr')

    train_generator = data_generator(data_source = data,
                                     label_source = label,
                                     batch_size = batch_size)
    valid_generator = data_generator(data_source = data_valid, 
                                     label_source = label_valid,
                                     batch_size = batch_size)
    
    input_shape = data_shape_extractor(data, batch_size)
    print(input_shape)
    
    model = resnet_model(input_shape,num_classes=num_classes)
    # model = transformer_model(input_shape, num_heads=8, ff_dim=256, num_layers=2, num_classes=num_classes,d_model=256)
    # model = lstm_model(input_shape,neurons=128, addlayers=2,num_classes=num_classes,d_model=256,dropout=0.0)
    # model = resnet_model_spec(input_shape,num_classes=num_classes)


    early_stop = EarlyStopping('val_loss', min_delta=0, patience=10) # Modify as needed
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=5, verbose=1) # Modify as needed
    callbacks = [early_stop, reduce_lr] # Modify as needed

    opt = RMSprop(learning_rate=1e-3) # Modify as needed

    model.compile(loss=['categorical_crossentropy'], optimizer=opt)

    history = model.fit(train_generator,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_generator,
                        validation_steps=num_valid_samples // batch_size,
                        epochs=1000, 
                        verbose=1,
                        callbacks=callbacks)
    

    timestamp1 = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_png1 = f'learning_curve_{timestamp1}.png'
    save_dir = './graphs/'
    file_path_png1 = os.path.join(save_dir, filename_png1)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Learning Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.grid(True)
    plt.savefig(file_path_png1, format='png', bbox_inches='tight')
    print(f"Learning curve saved as: {file_path_png1}")

    return model
    
    

def inference(data_path, tx_range, model_path,pk_range):

    model = tf.keras.models.load_model(model_path, compile=False)

    data, label = load_iq(data_path,tx_range,pk_range)
    label = label -1

    ###### only applies to spectrogram inputs ######

    # data = channel_ind_spectrogram(data)
    # data = data[:, :, :, 0] # only for transformer & lstm
    # data = data.transpose(0, 2, 1) # only for transformer & lstm

    ################################################

    pred_prob = model.predict(data)

    pred_label = pred_prob.argmax(axis=-1)
    #conf_mat = confusion_matrix(label, pred_label)
    acc = accuracy_score(label, pred_label)
    print('Overall accuracy = %.4f' % acc)

    return acc


if __name__ == '__main__':

    packet_size_test_range = ['76900'] #enter packet size(s)
    train_days = ['d1']
    eval_days = ['d1','d2']
    overall_ave_acc = []

    for size in packet_size_test_range:
        for trainday in train_days:
            for runday in eval_days:
                legitdevs = 3 # legitimate devices - modify as needed
                trainset_name = f'./dataset/{trainday}_{size}_trainingset.h5' #enter path to training and test set respectively
                testset_name = f'./dataset/{runday}_{size}_testset.h5'

                max_packs_train = 237
                max_packs_test = 34

                train_pkt_range = np.arange(0,max_packs_train, dtype = int)
                test_pkt_range = np.arange(0,max_packs_test, dtype = int)

                range_dev_train = np.arange(1,legitdevs+1, dtype = int)
                range_dev_clf = np.arange(1,legitdevs+1, dtype = int)


                ########### Model Setup#####################
                start_train = time.time()

                model_trained = train(data_path = trainset_name,tx_range = range_dev_train,pk_range = train_pkt_range)
                model_save_name = f'Model_train{trainday}_test{runday}_{size}.h5'
                tf.keras.models.save_model(model_trained, model_save_name)

                end_train = time.time()
                train_time = end_train-start_train

                start_clf = time.time()

                acc = inference(data_path=testset_name,tx_range=range_dev_clf,model_path=f'./{model_save_name}',pk_range=test_pkt_range)
                    
                end_clf = time.time()
                clf_time = end_clf - start_clf

                ########################################
                    

                overall_ave_acc.append([size,trainday,runday,acc,train_time, clf_time])
                        
    overall_acc_dir = './graphs/'
    print_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_overall_acc = f'Overall accuracies_{print_date}'
    filepath_overall_acc = os.path.join(overall_acc_dir, filename_overall_acc)

    with open(filepath_overall_acc,'w') as g:
        g.write('packet size, train day, test day, acc, train time, clf time \n')
        for line in overall_ave_acc:
            g.write(f"{line}\n")
                

