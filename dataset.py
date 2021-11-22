import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import random
from tqdm import tqdm
import torch
import librosa
import torchaudio

class RespiDataset(Dataset):
    def __init__(self, split, initialize=True, data_dir="dataset/spec_cut", num_mel=128, multi_label=False, mean=None, std=None, fixed_length=None):
        super(RespiDataset, self).__init__()
        self.split=split
        assert self.split in ['train', 'val'], "split must be either train or val"
        if self.split == 'train':
            self.train_data = []
        else:
            self.val_data = []
        self.data_dir=data_dir
        self.path=os.listdir(self.data_dir)
        # only used if data need to be in fixed length
        self.fixed_length=fixed_length
        self.multi_label = multi_label
        self.num_mel = num_mel
        self.weights = []
        if initialize:
            if mean is None or std is None:
                self.mean, self.std = self.initialize(self.path, self.multi_label)
            else:
                self.mean = mean
                self.std = std
            print(self.mean, self.std)
        else:
            self.mean = mean
            self.std = std

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((self.num_mel, self.fixed_length))
            ])
        
        self.fm = torchaudio.transforms.FrequencyMasking(24)
        self.tm = torchaudio.transforms.TimeMasking(48)
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            # breakpoint()
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    # temp_wav = torch.zeros(1, waveform1.shape[1])
                    # temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    # waveform2 = temp_wav
                    # duplicating
                    temp_wav = waveform2.repeat(1, waveform1.shape[-1]//waveform2.shape[-1] + 1)
                    waveform2 = temp_wav[0, 0:waveform1.shape[-1]]
                else:
                    # front cutting
                    # waveform2 = waveform2[0, 0:waveform1.shape[1]]
                    # random cutting
                    randidx = np.random.randint(low=0, high=waveform2.shape[1]-waveform1.shape[1], size=(1,))
                    waveform2 = waveform2[0, randidx[0]:randidx[0]+waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        fbank=fbank.permute(1,0).unsqueeze(0)

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < 0.5 and self.split == 'train':
            datum = self.train_data[index]
            mix_sample_idx = random.randint(0, len(self.train_data)-1)
            mix_datum = self.train_data[mix_sample_idx]

            fbank, mix_lambda = self._wav2fbank(datum, mix_datum)
            # initialize the label
            label1 = torch.from_numpy(np.array(self.labels[index]))
            label2 = torch.from_numpy(np.array(self.labels[mix_sample_idx]))
            label_indices = label1 * mix_lambda + label2 * (1.0-mix_lambda)
        # if not do mixup
        else:
            if self.split == 'train':
                datum = self.train_data[index]
            else:
                datum = self.val_data[index]
            fbank, mix_lambda = self._wav2fbank(datum)
            label_indices = torch.from_numpy(np.array(self.labels[index]))

        # normalize the input
        if fbank.shape[-1] < self.fixed_length:
            fbank = fbank.repeat(1, 1, self.fixed_length//fbank.shape[-1] + 1)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.split == 'train':
            fbank = self.transforms(fbank)
            fbank = self.tm(fbank)
            fbank = self.fm(fbank)
        else:
            fbank = fbank[:,:,:self.fixed_length]

        return fbank, label_indices.long()


    def initialize(self, paths, multi_label):
        wavs = [torch.empty(1)]*len(paths)
        labels = [np.empty(1)]*len(paths)
        for i, s in tqdm(enumerate(paths),total=len(paths)):
            sp = self.data_dir+"/"+s

            ann = s.split('_')[-1].split('.')[0]
            wavs[i] = sp
            if multi_label:
                ann = self.to_multi_hot(ann)
            else:
                # ann = self.to_int(ann, 2)
                # ann = self.to_multi_hot(ann)
                ann = self.to_one_hot(ann)
            labels[i] = ann
        self.data = wavs
        self.labels = labels
        return 0, 1
    

    def to_multi_hot(self, ann):
        label = [0]*len(ann)
        for i, an in enumerate(ann):
            if an == '1':
                label[i] = 1
        return label

    def to_one_hot(self, ann):
        label = [0]*(2**len(ann))
        label[int(ann,2)] = 1.0
        return label

    def to_int(self, ann):
        label = int(ann, 2)
        return label

    def __len__(self):
        return len(self.data)




class RespiWAVDataset(Dataset):
    def __init__(self, split, initialize=True, data_dir="dataset/spec_cut", num_mel=128, multi_label=False, mean=None, std=None, fixed_length=None):
        super(RespiWAVDataset, self).__init__()
        self.split=split
        assert self.split in ['train', 'val'], "split must be either train or val"
        if self.split == 'train':
            self.train_data = []
        else:
            self.val_data = []
        self.data_dir=data_dir
        self.path=os.listdir(self.data_dir)
        # only used if data need to be in fixed length
        self.fixed_length=fixed_length
        self.multi_label = multi_label
        if initialize:
            self.initialize(self.path, self.multi_label)
        else:
            pass
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((1, self.fixed_length))
            ])


    def __getitem__(self, i):
        # Get i-th path.
        if self.split == 'train':
            file = self.train_data[i]
        else:
            file = self.val_data[i]
        label= torch.from_numpy(np.array(self.labels[i]))
        wav = torch.from_numpy(file)
        wav = self.transforms(wav.unsqueeze(0))
        #print(spec.shape)
        return wav, label

    def initialize(self, paths, multi_label):
        wavs = [np.empty(1)]*len(paths)
        labels = [np.empty(1)]*len(paths)
        for i, s in tqdm(enumerate(paths),total=len(paths)):
            owav, _ = librosa.load(self.data_dir+"/"+s, sr=16000, res_type='kaiser_fast')
            if owav.shape[-1] < self.fixed_length:
                owav = owav.repeat(self.fixed_length//owav.shape[-1] + 1)
            ann = s.split('_')[-1].split('.')[0]
            wavs[i] = owav
            if multi_label:
                ann = self.to_multi_hot(ann)
            else:
                ann = int(ann, 2)
            labels[i] = ann
        self.data = wavs
        self.labels = labels
    

    def to_multi_hot(self, ann):
        label = [0]*len(ann)
        for i, an in enumerate(ann):
            if an == '1':
                label[i] = 1
        return label


    def __len__(self):
        return len(self.data)


class RespiDataset_Diag(Dataset):
    def __init__(self, split, initialize=True, data_dir="", num_mel=128, multi_label=False, mean=None, std=None, fixed_length=None):
        super(RespiDataset_Diag, self).__init__()
        self.idx2cls = ['Healthy', 'Asthma', 'Bronchiectasis', 'Bronchiolitis', 
        'COPD', 'LRTI', 'Pneumonia', 'URTI']
        self.cls2idx = {'Healthy':0, 'Asthma':1, 'Bronchiectasis':2, 'Bronchiolitis':3, 
        'COPD':4, 'LRTI':5, 'Pneumonia':6, 'URTI':7}
        self.diagdict = {'101': 'URTI', '102': 'Healthy', '103': 'Asthma', 
        '104': 'COPD', '105': 'URTI', '106': 'COPD', '107': 'COPD', '108': 'LRTI', 
        '109': 'COPD', '110': 'COPD', '111': 'Bronchiectasis', '112': 'COPD', 
        '113': 'COPD', '114': 'COPD', '115': 'LRTI', '116': 'Bronchiectasis', 
        '117': 'COPD', '118': 'COPD', '119': 'URTI', '120': 'COPD', '121': 'Healthy', 
        '122': 'Pneumonia', '123': 'Healthy', '124': 'COPD', '125': 'Healthy', 
        '126': 'Healthy', '127': 'Healthy', '128': 'COPD', '129': 'URTI', 
        '130': 'COPD', '131': 'URTI', '132': 'COPD', '133': 'COPD', '134': 'COPD', 
        '135': 'Pneumonia', '136': 'Healthy', '137': 'URTI', '138': 'COPD', 
        '139': 'COPD', '140': 'Pneumonia', '141': 'COPD', '142': 'COPD', 
        '143': 'Healthy', '144': 'Healthy', '145': 'COPD', '146': 'COPD', 
        '147': 'COPD', '148': 'URTI', '149': 'Bronchiolitis', '150': 'URTI', 
        '151': 'COPD', '152': 'Healthy', '153': 'Healthy', '154': 'COPD', 
        '155': 'COPD', '156': 'COPD', '157': 'COPD', '158': 'COPD', '159': 'Healthy', 
        '160': 'COPD', '161': 'Bronchiolitis', '162': 'COPD', '163': 'COPD', 
        '164': 'URTI', '165': 'URTI', '166': 'COPD', '167': 'Bronchiolitis', 
        '168': 'Bronchiectasis', '169': 'Bronchiectasis', '170': 'COPD', 
        '171': 'Healthy', '172': 'COPD', '173': 'Bronchiolitis', '174': 'COPD', 
        '175': 'COPD', '176': 'COPD', '177': 'COPD', '178': 'COPD', '179': 'Healthy', 
        '180': 'COPD', '181': 'COPD', '182': 'Healthy', '183': 'Healthy', 
        '184': 'Healthy', '185': 'COPD', '186': 'COPD', '187': 'Healthy', 
        '188': 'URTI', '189': 'COPD', '190': 'URTI', '191': 'Pneumonia', 
        '192': 'COPD', '193': 'COPD', '194': 'Healthy', '195': 'COPD', 
        '196': 'Bronchiectasis', '197': 'URTI', '198': 'COPD', '199': 'COPD', 
        '200': 'COPD', '201': 'Bronchiectasis', '202': 'Healthy', '203': 'COPD', 
        '204': 'COPD', '205': 'COPD', '206': 'Bronchiolitis', '207': 'COPD', 
        '208': 'Healthy', '209': 'Healthy', '210': 'URTI', '211': 'COPD', 
        '212': 'COPD', '213': 'COPD', '214': 'Healthy', '215': 'Bronchiectasis', 
        '216': 'Bronchiolitis', '217': 'Healthy', '218': 'COPD', '219': 'Pneumonia', 
        '220': 'COPD', '221': 'COPD', '222': 'COPD', '223': 'COPD', '224': 'Healthy', 
        '225': 'Healthy', '226': 'Pneumonia'}
        self.split=split
        assert self.split in ['train', 'val'], "split must be either train or val"
        self.data_dir=data_dir
        self.path=os.listdir(self.data_dir)
        # only used if data need to be in fixed length
        self.fixed_length=fixed_length
        self.multi_label = multi_label
        self.num_mel = num_mel
        self.weights = []
        if initialize:
            if mean is None or std is None:
                self.mean, self.std = self.initialize(self.path, self.multi_label)
            else:
                self.mean = mean
                self.std = std
            print(self.mean, self.std)
        else:
            self.mean = mean
            self.std = std

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((self.num_mel, self.fixed_length))
            ])
        
        self.fm = torchaudio.transforms.FrequencyMasking(24)
        self.tm = torchaudio.transforms.TimeMasking(48)
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            # breakpoint()
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    # temp_wav = torch.zeros(1, waveform1.shape[1])
                    # temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    # waveform2 = temp_wav
                    # duplicating
                    temp_wav = waveform2.repeat(1, waveform1.shape[-1]//waveform2.shape[-1] + 1)
                    waveform2 = temp_wav[0, 0:waveform1.shape[-1]]
                else:
                    # front cutting
                    # waveform2 = waveform2[0, 0:waveform1.shape[1]]
                    # random cutting
                    randidx = np.random.randint(low=0, high=waveform2.shape[1]-waveform1.shape[1], size=(1,))
                    waveform2 = waveform2[0, randidx[0]:randidx[0]+waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        fbank=fbank.permute(1,0).unsqueeze(0)

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < 0.5 and self.split == 'train':
            datum = self.data[index]
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            fbank, mix_lambda = self._wav2fbank(datum, mix_datum)
            # initialize the label
            label1 = torch.from_numpy(np.array(self.labels[index]))
            label2 = torch.from_numpy(np.array(self.labels[mix_sample_idx]))
            label_indices = label1 * mix_lambda + label2 * (1.0-mix_lambda)
        # if not do mixup
        else:
            datum = self.data[index]
            fbank, mix_lambda = self._wav2fbank(datum)
            label_indices = torch.from_numpy(np.array(self.labels[index]))

        # normalize the input
        if fbank.shape[-1] < self.fixed_length:
            fbank = fbank.repeat(1, 1, self.fixed_length//fbank.shape[-1] + 1)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.split == 'train':
            fbank = self.transforms(fbank)
            fbank = self.tm(fbank)
            fbank = self.fm(fbank)
        else:
            fbank = fbank[:,:,:self.fixed_length]

        return fbank, label_indices.long()



    def get_example(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        
        datum = self.data[index]
        mix_sample_idx = random.randint(0, len(self.data)-1)
        mix_datum = self.data[mix_sample_idx]

        fbank1 ,_ = self._wav2fbank(datum)
        fbank2 ,_ = self._wav2fbank(mix_datum)
        mixbank, mix_lambda = self._wav2fbank(datum, mix_datum)
        # initialize the label
        label1 = torch.from_numpy(np.array(self.labels[index]))
        label2 = torch.from_numpy(np.array(self.labels[mix_sample_idx]))
        label_indices = label1 * mix_lambda + label2 * (1.0-mix_lambda)

        return fbank1, fbank2, mixbank, label_indices


    def initialize(self, paths, multi_label):
        wavs = [torch.empty(1)]*len(paths)
        labels = [np.empty(1)]*len(paths)
        for i, s in tqdm(enumerate(paths),total=len(paths)):
            sp = self.data_dir+"/"+s

            patient = s.split('_')[0]
            wavs[i] = sp
            
            labels[i] = self.to_one_hot(self.cls2idx[self.diagdict[patient]])
        self.data = wavs
        self.labels = labels
        return 0, 1
    
    def to_one_hot(self, ann):
        label = [0]*8
        label[ann] = 1.0
        return label

    def __len__(self):
        return len(self.data)