import numpy as np
import h5py


from scipy import signal



class LoadDataset():
    def __init__(self,):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        
    
    def load_iq(self, file_path, dev_range, pkt_range):
        f = h5py.File(file_path,'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        
        sample_index_list = []
        
        for dev_idx in dev_range:
            sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
    
        data = f[self.dataset_name][sample_index_list]
        data_real_imag = np.stack((data.real, data.imag), axis=-1)
        
        label = label[sample_index_list]
          
        f.close()
        return data,label #return data_real_imag for slices



class ChannelIndSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self,data):
        s_norm = np.zeros(data.shape, dtype=complex)
        
        for i in range(data.shape[0]):
        
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i]/rms
        
        return s_norm        

    def _spec_crop(self, x):


        num_col = x.shape[1]
        start = round(num_col * 0.4)
        end = round(num_col * 0.6)
        x_cropped = x[:, start:end]
    
        return x_cropped


    def _gen_single_channel_ind_spectrogram(self, sig, win_len=256, overlap=128):
        f, t, spec = signal.stft(sig, 
                                window='boxcar', 
                                nperseg= win_len, 
                                noverlap= overlap, 
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)

        spec = np.fft.fftshift(spec, axes=0) # shift to adjust the central frequency

        chan_ind_spec = spec[:,1:]/spec[:,:-1] #isolate channel effects
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec)**2)
                  
        return chan_ind_spec_amp
    


    def channel_ind_spectrogram(self, data):
        data = self._normalization(data)
        

        num_sample = data.shape[0]
        num_row = int(200)
        win_len = num_row
        overlap = (win_len/8)*7


        sample_spec = self._gen_single_channel_ind_spectrogram(data[0], win_len, overlap)
        sample_spec = self._spec_crop(sample_spec)
        num_column = sample_spec.shape[1]  # Columns based on actual sample spectrogram output
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        for i in range(num_sample):
                   
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i],win_len,overlap)
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i,:,:,0] = chan_ind_spec_amp   
        return data_channel_ind_spec

