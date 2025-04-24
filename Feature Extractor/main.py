import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from datetime import datetime
import h5py
import time


from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop


from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from deep_learning_models import TripletNet, identity_loss, triplet_loss_fcn




def train_feature_extractor(file_path, pkt_range, dev_range):
    
    LoadDatasetObj = LoadDataset()
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    TripletNetObj = TripletNet()

    
    data, label = LoadDatasetObj.load_iq(file_path, dev_range, pkt_range)

    
    #############for spectrogram inputs only #######################
    spectros = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    spectros = spectros[:,:,:,0] # only for transformers and lstm spectrograms
    data = spectros.transpose(0,2,1) # only for transformers and lstm spectrograms

    ################################################################

    
    input_shape=data.shape
    print(input_shape)
    margin = 0.1
    batch_size = 20

    feature_extractor = TripletNetObj.feature_extractor_transformer(input_shape,num_heads=4, ff_dim=256, num_layers=2, output_dim=512,d_model=512,dropout=0.0)
    #feature_extractor = TripletNetObj.feature_extractor_lstm(input_shape,neurons=256, plusoneLSTM=1, output_dim=512,dropout=0.0)
    #feature_extractor = TripletNetObj.feature_extractor_resnet_spec(input_shape)
    #feature_extractor = TripletNetObj.feature_extractor_resnet_slice(input_shape)





    triplet_net = TripletNetObj.create_triplet_net(feature_extractor, input_shape, margin)

    early_stop = EarlyStopping('val_loss',min_delta = 0,patience = 20)
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta = 0, factor = 0.2, patience = 10, verbose=1)
    callbacks = [early_stop, reduce_lr]
    
    data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.1, shuffle= True)
    train_generator = TripletNetObj.create_generator(batch_size, dev_range, data_train,label_train)
    valid_generator = TripletNetObj.create_generator(batch_size, dev_range, data_valid, label_valid)


    opt = RMSprop(learning_rate=1e-4)
    #triplet_net.compile(loss = identity_loss, optimizer = opt) #for resnet only
    triplet_net.compile(loss = triplet_loss_fcn(margin=margin),optimizer=opt)

    history = triplet_net.fit(train_generator,
                              steps_per_epoch = data_train.shape[0]//batch_size,
                              epochs = 1000,
                              validation_data = valid_generator,
                              validation_steps = data_valid.shape[0]//batch_size,
                              verbose=1, 
                              callbacks = callbacks)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    timestamp1 = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_png1 = f'learning_curve_{timestamp1}.png'
    save_dir = './graphs/'
    file_path_png1 = os.path.join(save_dir, filename_png1)

    plt.figure()
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path_png1, format='png', bbox_inches='tight')
    print(f"Learning curve saved as: {file_path_png1}")
    
    return feature_extractor



def test_classification(file_path_enrol,file_path_clf,feature_extractor_name,pkt_range_enrol,pkt_range_clf,dev_range_enrol,dev_range_clf ):

   
    feature_extractor = tf.keras.models.load_model(f'./{feature_extractor_name}',compile=False)
    
    LoadDatasetObj = LoadDataset()
    

    data_enrol, label_enrol = LoadDatasetObj.load_iq(file_path_enrol, dev_range_enrol, pkt_range_enrol)
    
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    
    #############for spectrogram inputs only ##############
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)
    data_enrol = data_enrol[:, :, :, 0] # only for transformers and lstm spectrograms
    data_enrol = data_enrol.transpose(0,2,1) # only for transformers and lstm spectrograms
    #######################################################
    

    feature_enrol = feature_extractor.predict(data_enrol)
    del data_enrol
    
    knnclf=KNeighborsClassifier(n_neighbors=15,metric='euclidean')
    knnclf.fit(feature_enrol, np.ravel(label_enrol))
    
    
    
    data_clf, true_label = LoadDatasetObj.load_iq(file_path_clf, dev_range_clf, pkt_range_clf)
    
    #############for spectrogram inputs only ##############
    data_clf = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_clf)
    data_clf = data_clf[:, :, :, 0] # only for transformers and lstm spectrograms
    data_clf = data_clf.transpose(0,2,1) # only for transformers and lstm spectrograms
    #######################################################

    feature_clf = feature_extractor.predict(data_clf)
    del data_clf


    pred_label = knnclf.predict(feature_clf)

    acc = accuracy_score(true_label, pred_label)
    print('Overall accuracy = %.4f' % acc)
    
    return pred_label, true_label, acc



def test_rogue_device_detection(feature_extractor_name,file_path_enrol,dev_range_enrol,pkt_range_enrol,file_path_legitimate,dev_range_legitimate,pkt_range_legitimate,file_path_rogue,dev_range_rogue,pkt_range_rogue):

    def _compute_eer(fpr,tpr,thresholds): #EER = Equal Error Rate

        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        return eer, thresholds[min_index]
    

    feature_extractor = tf.keras.models.load_model(f"./{feature_extractor_name}",compile=False)
    
    LoadDatasetObj = LoadDataset()
    

    data_enrol, label_enrol = LoadDatasetObj.load_iq(file_path_enrol, dev_range_enrol, pkt_range_enrol)
    
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    

 
    #############for spectrogram inputs only ##############
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)
    data_enrol = data_enrol[:, :, :, 0] # only for transformers and lstm spectrograms
    data_enrol = data_enrol.transpose(0,2,1) # only for transformers and lstm spectrograms
    #######################################################
    

    feature_enrol = feature_extractor.predict(data_enrol)
    del data_enrol
    
    knnclf=KNeighborsClassifier(n_neighbors=15,metric='euclidean')
    knnclf.fit(feature_enrol, np.ravel(label_enrol))

    data_legitimate, label_legitimate = LoadDatasetObj.load_iq(file_path_legitimate, dev_range_legitimate, pkt_range_legitimate)

    data_rogue, label_rogue = LoadDatasetObj.load_iq(file_path_rogue, dev_range_rogue, pkt_range_rogue)
    

    # combine legit and rogue dataset
    data_test = np.concatenate([data_legitimate,data_rogue]) 
    label_test = np.concatenate([label_legitimate,label_rogue])
    
    #############for spectrogram inputs only ##############
    data_test = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)
    data_test = data_test[:, :, :, 0] # only for transformers and lstm spectrograms
    data_test = data_test.transpose(0,2,1) # only for transformers and lstm spectrograms
    #######################################################


    feature_test = feature_extractor.predict(data_test)
    del data_test

    distances, indexes = knnclf.kneighbors(feature_test)
    detection_score = distances.mean(axis =1)

    true_label = np.zeros([len(label_test),1]) #legit devices are labelled 1, rogue are labelled 0
    

    true_label[(label_test <= dev_range_legitimate[-1]) & (label_test >= dev_range_legitimate[0])] = 1
    
    fpr, tpr, thresholds = roc_curve(true_label, detection_score, pos_label = 1)
    
    fpr = 1-fpr  
    tpr = 1-tpr

    eer, _ = _compute_eer(fpr,tpr,thresholds)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, eer
    


if __name__ == '__main__':

    packet_size_test_range = ['76900']
    data_day = ['d1','d2']
    overall_stats = []

    for size in packet_size_test_range:
        for trainday in data_day:
            for runday in data_day:
                builds = 3
                legitdevs = 3
                roguedevs = 1
                trainset_name = f'./dataset/{trainday}_{size}_trainingset.h5'
                enrolset_name = f'./dataset/{runday}_{size}_enrollmentset.h5'
                testset_name = f'./dataset/{runday}_{size}_testset.h5'
                legitset_name = f'./dataset/{runday}_{size}_legitset.h5'
                rogueset_name = f'./dataset/{runday}_{size}_rogueset.h5'
                
                
                max_packs_train = 237
                max_packs_enroll = 69
                max_packs_test = 34
                max_packs_legit = 34
                max_packs_rogue = 34

                train_pkt_range = np.arange(0,max_packs_train, dtype = int)
                enrol_pkt_range = np.arange(0,max_packs_enroll, dtype = int)
                test_pkt_range = np.arange(0,max_packs_test, dtype = int)
                legit_pkt_range = np.arange(0,max_packs_legit,dtype = int)
                rogue_pkt_range = np.arange(0,max_packs_rogue,dtype = int)

                range_dev_train = np.arange(1,legitdevs+1, dtype = int)
                range_dev_enrol = np.arange(1,legitdevs+1, dtype = int)
                range_dev_clf = np.arange(1,legitdevs+1, dtype = int)
                range_dev_legit = np.arange(1,legitdevs+1, dtype = int)
                range_dev_rogue = np.arange(legitdevs+1,legitdevs+1+roguedevs, dtype = int)

                ############Training################
                start_train = time.time()
                feature_extractor = train_feature_extractor(file_path = trainset_name,pkt_range = train_pkt_range,dev_range = range_dev_train)
                    
                extractor_save_name = f'Extractor_train{trainday}_test{runday}_{size}.h5'
                tf.keras.models.save_model(feature_extractor, extractor_save_name)

                end_train = time.time()
                train_time = end_train-start_train

                ##################################


                ############Classification###############

                start_clf = time.time()
                    
                pred_label, true_label, acc = test_classification(file_path_enrol = enrolset_name, 
                                                                      file_path_clf = testset_name,
                                                                      feature_extractor_name = extractor_save_name,
                                                                      pkt_range_enrol = enrol_pkt_range,
                                                                      pkt_range_clf = test_pkt_range,
                                                                      dev_range_enrol = range_dev_enrol,
                                                                      dev_range_clf = range_dev_clf)
                    
                end_clf = time.time()
                clf_time = end_clf - start_clf

                #conf_mat = confusion_matrix(true_label, pred_label)

                ############Rogue Det###################
                start_rogue = time.time()
                fpr, tpr, roc_auc, eer = test_rogue_device_detection(feature_extractor_name=extractor_save_name,
                                                                         file_path_enrol=enrolset_name,
                                                                         dev_range_enrol=range_dev_enrol,
                                                                         pkt_range_enrol=enrol_pkt_range,
                                                                         file_path_legitimate=legitset_name,
                                                                         dev_range_legitimate=range_dev_legit,
                                                                         pkt_range_legitimate=legit_pkt_range,
                                                                         file_path_rogue=rogueset_name,
                                                                         dev_range_rogue=range_dev_rogue,
                                                                         pkt_range_rogue=rogue_pkt_range)
                    
                end_rogue = time.time()
                rogue_time = end_rogue-start_rogue

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_dir = './graphs/'
                filename_roc = f'ROC_{timestamp}.png'
                file_path_roc = os.path.join(save_dir, filename_roc)
                plt.figure()
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr, tpr, label='Feature Extractor, AUC = ' + str(round(roc_auc,3)) + ', EER = ' + str(round(eer,3)), color='r')
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.legend(loc=4)
                plt.savefig(file_path_roc, format='png', bbox_inches='tight')
                ######################################
                
                overall_stats.append([size,trainday,runday,acc,roc_auc, eer, train_time, clf_time,rogue_time])
                

    overall_acc_dir = './graphs/'
    print_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_overall_acc = f'Overall accuracies_{print_date}'
    filepath_overall_acc = os.path.join(overall_acc_dir, filename_overall_acc)

    with open(filepath_overall_acc,'w') as g:
        g.write('packet size, train day, test day, acc,roc_auc, eer, train_time, clf_time,rogue_time \n')
        for line in overall_stats:
            g.write(f"{line}\n")


