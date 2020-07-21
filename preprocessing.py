import numpy as np
import scipy
import librosa
import torch
import MRCG as mrcg
from tqdm import tqdm



sr = 16000
frame_length = 0.025  # 25ms
frame_stride = 0.01  # 10ms
n_fft = int(round(sr * frame_length))
hop_length = int(round(sr * frame_stride))


def get_mag_phase(file_path, sr=sr, n_fft=n_fft, hop_length=hop_length, mono=True):
    audio, sr = librosa.load(file_path, sr=sr, mono=mono)

    if(mono == True):
        D = librosa.stft(np.asfortranarray(audio), n_fft=n_fft, hop_length=hop_length)
        D_mag, D_phase = librosa.magphase(D)

        avg = D_mag.mean()
        stdv = D_mag.std()
        D_mag = (D_mag - avg) / stdv
        return D_mag[1:, :], np.angle(D_phase)[1:, :]
    elif(mono == False):
        DL = librosa.stft(np.asfortranarray(audio[0]), n_fft=n_fft, hop_length=hop_length)
        DL_mag, DL_phase = librosa.magphase(DL)

        DR = librosa.stft(np.asfortranarray(audio[1]), n_fft=n_fft, hop_length=hop_length)
        DR_mag, DR_phase = librosa.magphase(DR)

        avg = DL_mag.mean()
        stdv = DL_mag.std()
        DL_mag = (DL_mag - avg) / stdv
        DR_mag = (DR_mag - avg) / stdv
        return (DL_mag[1:, :], np.angle(DL_phase)[1:, :]), (DR_mag[1:, :], np.angle(DR_phase)[1:, :])


def get_spectrogram(file_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, exp_b=0.3, mono=True):
    audio, sr = librosa.load(file_path, sr=sr, mono=mono)

    if(mono == True):
        stft_matrix = librosa.stft(np.asfortranarray(audio), n_fft=n_fft, hop_length=hop_length)
        mag_D = np.abs(stft_matrix)
        pwr = mag_D ** 2

        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_pwr = np.dot(mel_basis, pwr)
        return mel_pwr ** exp_b
    elif(mono == False):
        stft_L_matrix = librosa.stft(np.asfortranarray(audio[0]), n_fft=n_fft, hop_length=hop_length)
        mag_DL = np.abs(stft_L_matrix)
        pwr_L = mag_DL ** 2

        stft_R_matrix = librosa.stft(np.asfortranarray(audio[1]), n_fft=n_fft, hop_length=hop_length)
        mag_DR = np.abs(stft_R_matrix)
        pwr_R = mag_DR ** 2

        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_pwr_L = np.dot(mel_basis, pwr_L)
        mel_pwr_R = np.dot(mel_basis, pwr_R)
        return mel_pwr_L ** exp_b, mel_pwr_R ** exp_b


def get_cepstrogram(spec, dct_type=2, norm='ortho'):
    ### return the cepstrogram according to the spectrogram
    mel_ceps_coef = scipy.fftpack.dct(spec, axis=0, type=dct_type, norm=norm)
    mel_ceps_coef_relu = np.maximum(mel_ceps_coef, 0.0)
    return mel_ceps_coef_relu


def get_mfcc(file_path, sr=sr, n_mfcc=128, n_fft=n_fft , hop_length=hop_length, mono=True):
    audio, sr = librosa.load(file_path, sr=sr, mono=mono)
    
    if(mono == True):
        mfcc_matrix = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft , hop_length=hop_length)
        
        avg = mfcc_matrix.mean()
        stdv = mfcc_matrix.std()
        mfcc_matrix = (mfcc_matrix - avg) / stdv
        return mfcc_matrix
    elif(mono == False):
        mfcc_L = librosa.feature.mfcc(y=audio[0], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft , hop_length=hop_length)
        mfcc_R = librosa.feature.mfcc(y=audio[1], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft , hop_length=hop_length)
        
        avg = mfcc_L.mean()
        stdv = mfcc_L.std()
        mfcc_L = (mfcc_L - avg) / stdv
        mfcc_R = (mfcc_R - avg) / stdv
        return mfcc_L, mfcc_R


def get_mrcg(file_path, sr=sr, mono=True):
    audio, sr = librosa.load(file_path, sr=sr, mono=mono)

    if(mono == True):
        mrcg_matrix = mrcg.mrcg_extract(np.asfortranarray(audio), sr)
        return mrcg_matrix
    elif(mono == False):
        mrcg_L = mrcg.mrcg_extract(np.asfortranarray(audio[0]), sr)
        mrcg_R = mrcg.mrcg_extract(np.asfortranarray(audio[1]), sr)
        return mrcg_L, mrcg_R


def generatio_tensor_instances(array_2d, seq_len=50, hop=10, label=None):
    row_size, col_size = array_2d.shape[0], array_2d.shape[1]
    stack_array = []

    j = 0
    while(j <= (col_size - (seq_len + 1))):
        context_frame = array_2d[:, j:(j+seq_len)]
        stack_array.append(context_frame[..., np.newaxis])
            
        j += hop
    
    if label is not None:
        return np.stack(stack_array, axis=0), np.repeat(label, len(stack_array))
    else:
        return np.stack(stack_array, axis=0)


def preprocessing(data_path, method, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, n_mfcc=128, exp_b=0.3, seq_len=50, hop=10, mono=True, label=None):
    if label is not None:
        if(mono == True):
            if(method == 'stft'):
                mag, phase = get_mag_phase(data_path, sr, n_fft, hop_length, mono)
                mag_instances_sub, label = generatio_tensor_instances(mag, label=label)
                phase_instances_sub, _ = generatio_tensor_instances(phase, label=label)
                concat_tensor = np.concatenate([mag_instances_sub, phase_instances_sub], axis=-1)
                return concat_tensor, label
            elif(method == 'mel'):
                mel = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                mel_instances_sub, label = generatio_tensor_instances(mel, label=label)
                return mel_instances_sub, label
            elif(method == 'mfcc'):
                #mel = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                #mfcc = get_cepstrogram(mel, dct_type=2, norm='ortho')
                #mfcc_instances_sub, label = generatio_tensor_instances(mfcc, label=label)
                mfcc = get_mfcc(data_path, sr, n_mfcc, n_fft, hop_length, mono)
                mfcc_instances_sub, label = generatio_tensor_instances(mfcc, label=label)
                return mfcc_instances_sub, label
            elif(method == 'mrcg'):
                mrcg_matrix = get_mrcg(data_path, sr, mono)
                mrcg_instances_sub, label = generatio_tensor_instances(mrcg_matrix, label=label)
                return mrcg_instances_sub, label
        elif(mono == False):
            if(method == 'stft'):
                (mag_L, phase_L), (mag_R, phase_R) = get_mag_phase(data_path, sr, n_fft, hop_length, mono)
                mag_L_instances_sub, label = generatio_tensor_instances(mag_L, label=label)
                mag_R_instances_sub, _ = generatio_tensor_instances(mag_R, label)
                phase_L_instances_sub, _ = generatio_tensor_instances(phase_L, label=label)
                phase_R_instances_sub, _ = generatio_tensor_instances(phase_R, label=label)
                concat_tensor = np.concatenate([mag_L_instances_sub, mag_R_instances_sub, phase_L_instances_sub, phase_R_instances_sub],
                                                axis=-1)
                return concat_tensor, label
            elif(method == 'mel'):
                mel_L, mel_R = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                mel_L_instances_sub, label = generatio_tensor_instances(mel_L, label=label)
                mel_R_instances_sub, _ = generatio_tensor_instances(mel_R, label=label)
                concat_tensor = np.concatenate([mel_L_instances_sub, mel_R_instances_sub],
                                                axis=-1)
                return concat_tensor, label
            elif(method == 'mfcc'):
                mel_L, mel_R = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                mfcc_L = get_cepstrogram(mel_L, dct_type=2, norm='ortho')
                mfcc_R = get_cepstrogram(mel_R, dct_type=2, norm='ortho')
                mfcc_L_instances_sub, label = generatio_tensor_instances(mfcc_L, label=label)
                mfcc_R_instances_sub, _ = generatio_tensor_instances(mfcc_R, label=label)
                concat_tensor = np.concatenate([mfcc_L_instances_sub, mfcc_R_instances_sub],
                                                axis=-1)
                return concat_tensor, label
            elif(method == 'mrcg'):
                mrcg_L, mrcg_R = get_mrcg(data_path, sr, mono)
                mrcg_L_instances_sub, label = generatio_tensor_instances(mrcg_L, label=label)
                mrcg_R_instances_sub, _ = generatio_tensor_instances(mrcg_R, label=label)
                concat_tensor = np.concatenate([mrcg_L_instances_sub, mrcg_R_instances_sub],
                                                axis=-1)
                return concat_tensor, label
    else:
        if(mono == True):
            if(method == 'stft'):
                mag, phase = get_mag_phase(data_path, sr, n_fft, hop_length, mono)
                mag_instances_sub = generatio_tensor_instances(mag)
                phase_instances_sub = generatio_tensor_instances(phase)
                concat_tensor = np.concatenate([mag_instances_sub, phase_instances_sub], axis=-1)
                return concat_tensor
            elif(method == 'mel'):
                mel = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                mel_instances_sub = generatio_tensor_instances(mel)
                return mel_instances_sub
            elif(method == 'mfcc'):
                #mel = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                #mfcc = get_cepstrogram(mel, dct_type=2, norm='ortho')
                #mfcc_instances_sub = generatio_tensor_instances(mfcc)
                mfcc = get_mfcc(data_path, sr, n_mfcc, n_fft, hop_length, mono)
                mfcc_instances_sub = generatio_tensor_instances(mfcc)
                return mfcc_instances_sub
            elif(method == 'mrcg'):
                mrcg_matrix = get_mrcg(data_path, sr, mono)
                mrcg_instances_sub = generatio_tensor_instances(mrcg_matrix)
                return mrcg_instances_sub
        elif(mono == False):
            if(method == 'stft'):
                (mag_L, phase_L), (mag_R, phase_R) = get_mag_phase(data_path, sr, n_fft, hop_length, mono)
                mag_L_instances_sub = generatio_tensor_instances(mag_L)
                mag_R_instances_sub = generatio_tensor_instances(mag_R)
                phase_L_instances_sub = generatio_tensor_instances(phase_L)
                phase_R_instances_sub = generatio_tensor_instances(phase_R)
                concat_tensor = np.concatenate([mag_L_instances_sub, mag_R_instances_sub, phase_L_instances_sub, phase_R_instances_sub],
                                                axis=-1)
                return concat_tensor
            elif(method == 'mel'):
                mel_L, mel_R = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                mel_L_instances_sub = generatio_tensor_instances(mel_L)
                mel_R_instances_sub = generatio_tensor_instances(mel_R)
                concat_tensor = np.concatenate([mel_L_instances_sub, mel_R_instances_sub],
                                                axis=-1)
                return concat_tensor
            elif(method == 'mfcc'):
                mel_L, mel_R = get_spectrogram(data_path, sr, n_fft, hop_length, n_mels, exp_b, mono)
                mfcc_L = get_cepstrogram(mel_L, dct_type=2, norm='ortho')
                mfcc_R = get_cepstrogram(mel_R, dct_type=2, norm='ortho')
                mfcc_L_instances_sub = generatio_tensor_instances(mfcc_L)
                mfcc_R_instances_sub = generatio_tensor_instances(mfcc_R)
                concat_tensor = np.concatenate([mfcc_L_instances_sub, mfcc_R_instances_sub],
                                                axis=-1)
                return concat_tensor
            elif(method == 'mrcg'):
                mrcg_L, mrcg_R = get_mrcg(data_path, sr, mono)
                mrcg_L_instances_sub = generatio_tensor_instances(mrcg_L)
                mrcg_R_instances_sub = generatio_tensor_instances(mrcg_R)
                concat_tensor = np.concatenate([mrcg_L_instances_sub, mrcg_R_instances_sub],
                                                axis=-1)
                return concat_tensor

        
def convert_spectrograms(data_dirs, conv_dim, method='mfcc', sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, n_mfcc=128, exp_b=0.3, seq_len=50, hop=10, mono=True, labels=None):
    concat_tensors = []
    if labels is not None:
        concat_labels = []

    for i, data_dir in tqdm(enumerate(data_dirs)):
        try:
            if labels is not None:
                if(conv_dim == '1d'):
                    concat_tensor, label = preprocessing(data_dir, method=method, sr=sr, n_mels=40, n_mfcc=40, label=labels[i])
                elif(conv_dim == '2d'):
                    concat_tensor, label = preprocessing(data_dir, method=method, sr=sr, label=labels[i])
                concat_tensors.append(concat_tensor)
                concat_labels.append(label)
            else:
                if(conv_dim == '1d'):
                    concat_tensor = preprocessing(data_dir, method=method, sr=sr, n_mels=40, n_mfcc=40)
                elif(conv_dim == '2d'):
                    concat_tensor = preprocessing(data_dir, method=method, sr=sr)
                concat_tensors.append(concat_tensor)
        except:
            pass
        
    if labels is not None:
        spectrograms = np.concatenate(np.array(concat_tensors), axis=0)
        labels = np.concatenate(np.array(concat_labels), axis=0)
        return spectrograms, labels
    else:
        spectrograms = np.concatenate(np.array(concat_tensors), axis=0)
        return spectrograms


def convert_tensor(X, y=None):
    X = torch.tensor(X).float()
    X = X.permute(0, 3, 1, 2)
    if y is not None:
        y = torch.tensor(y).float()
        return X, y
    else:
        return X