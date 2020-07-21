import os
import glob
import argparse

import librosa
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-wav_data_dir", default='./wav_data/pretrain/RAVDESS_resample', type=str,
                        help="The wav data directory before add noise.")
    parser.add_argument("-output_dir", default='./wav_data/pretrain/RAVDESS_noise', type=str,
                        help="The wav data directory after add noise.")

    args = parser.parse_args()
    
    
    samples_dir = glob.glob(os.path.join(args.wav_data_dir, '**', '*.wav'), recursive=True)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    def add_noise(sample_dir, output_dir):
        y, sr = librosa.load(sample_dir, sr=16000)
        noise_amp = 0.05 * np.random.uniform() * np.amax(y)
        y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
        filename = sample_dir.split('/')[-1].split('.')[0] + '_noise.wav'
        filename = os.path.join(output_dir, filename)
        write(filename, sr, y_noise)


    for sample_dir in tqdm(samples_dir):
        add_noise(sample_dir, args.output_dir)


if __name__ == "__main__":
    main()
    