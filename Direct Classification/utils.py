import numpy as np
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform, randn
import h5py
from scipy import signal
import math 
import random

def shuffle(data, label):

    index = np.arange(len(data))
    np.random.shuffle(index)
    data = data[index]
    label = label[index]

    return data, label



def load_iq(file_path, dev_range, pkt_range):
    
    f = h5py.File(file_path,'r')
    label = f['label'][:]
    label = label.astype(int)

        
    sample_index_list = []
        
    for dev_idx in dev_range:
        sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)
    
    data = f['data'][sample_index_list]
        
    label = label[sample_index_list]

    data_real_imag = np.stack((data.real, data.imag), axis=-1) # only applies to slices
          
    f.close()
    return data_real_imag,label # for spectrograms return data



def data_generator(data_source, label_source, batch_size):
    
    while True:
        
        data = data_source
        label = label_source
        
        
        sample_ind = random.sample(range(0, len(data)), batch_size)
        
        data = data[sample_ind]        
        label = label[sample_ind]

        ####### only applies to spectrogram inputs ######

        # data = channel_ind_spectrogram(data)
        # data = data[:,:,:,0] # only for transformer & lstm
        # data = data.transpose(0,2,1) # only for transformer & lstm

        #################################################
        
        yield data, label

def data_shape_extractor(data_source, batch_size):
    
    data = data_source
    sample_ind = random.sample(range(0, len(data)), batch_size)
    data = data[sample_ind]

    ####### only applies to spectrogram inputs ######

    # data = channel_ind_spectrogram(data)
    # data = data[:,:,:,0] # only for transformer & lstm
    # data = data.transpose(0,2,1) # only for transformer & lstm

    #################################################
    
    
    return data.shape

def normalization(data):
    s_norm = np.zeros(data.shape, dtype=complex)
        
    for i in range(data.shape[0]):
        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude**2))
        s_norm[i] = data[i]/rms
        
    return s_norm

def spec_crop(x):

    num_col = x.shape[1]
    start = round(num_col * 0.3) #set custom bounds
    end = round(num_col * 0.7) #set custom bounds
    x_cropped = x[:, start:end]
    
    return x_cropped

def gen_single_channel_ind_spectrogram(sig, win_len=200, overlap=175):
        
       # Short-time Fourier Transform 
        f, t, spec = signal.stft(sig, # IQ signal
                                window='boxcar', 
                                nperseg= win_len, # window lenght
                                noverlap= overlap, # overlap
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)
        
        spec = np.fft.fftshift(spec, axes=0) # shift to adjust the central frequency
        
        chan_ind_spec = spec[:,1:]/spec[:,:-1] # cancel channel effects
           
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec)**2) #log of magnitude values
                  
        return chan_ind_spec_amp

def channel_ind_spectrogram(data):

        data = normalization(data)
        num_sample = data.shape[0]
        num_row = int(200) # modify as needed
        win_len = num_row
        overlap = (win_len/8)*7 # modify as needed

        sample_spec = gen_single_channel_ind_spectrogram(data[0], win_len, overlap) # generate a sample spectrogram to get final shape
        sample_spec = spec_crop(sample_spec) # crop a sample spectrogram to get final shape
        num_column = sample_spec.shape[1]  # columns based on actual spectrogram output
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
                   
            chan_ind_spec_amp = gen_single_channel_ind_spectrogram(data[i],win_len,overlap)
            chan_ind_spec_amp = spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i,:,:,0] = chan_ind_spec_amp
            
        return data_channel_ind_spec

