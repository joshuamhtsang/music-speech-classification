import librosa
import librosa.display
import numpy as np
import argparse
from tqdm import tqdm
import os
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', help='Directory containing wav files', required=True)
    parser.add_argument('-ot', '--output_directory_tensors', help='Directory to save Mel Spectrogram tensors.',  required=True)
    parser.add_argument('-oimg', '--output_directory_images', help='Directory to save Mel Spectrogram images.',  required=True)
    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory_tensors = args.output_directory_tensors
    output_directory_images = args.output_directory_images

    if not os.path.exists(output_directory_tensors):
        os.makedirs(output_directory_tensors)

    if not os.path.exists(output_directory_images):
        os.makedirs(output_directory_images)

    for file in tqdm(os.listdir(input_directory)):
        y, sr = librosa.load('{}/{}'.format(input_directory, file))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        #S = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        print(np.shape(S))
        print(np.max(S))
        #S = librosa.power_to_db(S, ref=np.max)
        #S /= np.max(np.abs(S),axis=0)
        np.save('{}/{}'.format(output_directory_tensors, file), S)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S, y_axis='mel', fmax=8000, x_axis='time')
        #librosa.display.specshow(S, y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        #plt.show()
        plt.savefig('./{}/{}.png'.format(output_directory_images, file))
        plt.close()
