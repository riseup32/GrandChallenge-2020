import os
import glob
import argparse

import wave
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-pcm_data_dir", default='./wav_data/pretrain/KsponSpeech', type=str,
                        help="The pcm data directory.")
    parser.add_argument("-output_dir", default='./wav_data/pretrain/KsponSpeech_wav', type=str,
                        help="The wav data directory.")

    args = parser.parse_args()


    file_paths = glob.glob(os.path.join(args.pcm_data_dir, '**', '*.pcm'), recursive=True)
    file_paths = sorted(file_paths)


    def pcm2wav(pcm_file, output_dir=args.output_dir, channels=1, bit_depth=16, sampling_rate=16000):
        if(bit_depth % 8 != 0):
            raise ValueError("bit_depth ") + str(bit_depth) + "must be a multiple of 8."

        file_name = pcm_file.split('/')[-1].split('.')[0] + '.wav'
        wav_file = os.path.join(output_dir, file_name)

        with open(pcm_file, 'rb') as opened_pcm_file:
            pcm_data = opened_pcm_file.read()

            obj2write = wave.open(wav_file, 'wb')
            obj2write.setnchannels(channels)
            obj2write.setsampwidth(bit_depth // 8)
            obj2write.setframerate(sampling_rate)
            obj2write.writeframes(pcm_data)
            obj2write.close()


    for file_path in tqdm(file_paths):
        pcm2wav(file_path)


if __name__ == "__main__":
    main()
