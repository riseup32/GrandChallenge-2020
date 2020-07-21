import os
import glob
import argparse

import librosa
from tqdm import tqdm
from scipy.io.wavfile import write


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-wav_data_dir", default='./wav_data/pretrain/RAVDESS', type=str,
                        help="The wav data directory before resampling.")
    parser.add_argument("-output_dir", default='./wav_data/pretrain/RAVDESS_resample', type=str,
                        help="The wav data directory after resampling.")
    parser.add_argument("-orig_sr", default=48000, type=int,
                        help="The original sampling rate of y.")
    parser.add_argument("-target_sr", default=16000, type=int,
                        help="The target sampling rate.")

    args = parser.parse_args()


    samples_dir = glob.glob(os.path.join(args.wav_data_dir, '**', '*.wav'), recursive=True)


    def wav_resample(sample_dir, output_dir, orig_sr, target_sr):
        y, sr = librosa.load(sample_dir, orig_sr)
        y_resample = librosa.resample(y, sr, target_sr)
        filename = os.path.join(output_dir, sample_dir.split('/')[-1])
        write(filename, target_sr, y_resample)


    for sample_dir in tqdm(samples_dir):
        wav_resample(sample_dir, args.output_dir, args.orig_sr, args.target_sr)


if __name__ == "__main__":
    main()
    