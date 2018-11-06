import os
import argparse
from tqdm import tqdm
from random import shuffle

import numpy as np
from sklearn.utils import shuffle
import keras as ks
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras import backend as k

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_tensors(ids, tensor_directory_path, label):
    #feature_tensors = np.zeros( (len(ids), 128, 1292) )
    feature_tensors = np.zeros( (len(ids), 1292, 128) )
    label_tensors = np.zeros( (len(ids), 1) )

    for i, id in tqdm(enumerate(ids), total=len(ids), desc='Loading tensors'):
        feature_tensor = np.load('{}/{}.wav.npy'.format(tensor_directory_path, id)).astype('float32')
        print(id + str(np.shape(feature_tensor)))
        #feature_tensors[i] = feature_tensor
        feature_tensors[i] = np.transpose(feature_tensor)
        label_tensors[i] = label

    return feature_tensors, label_tensors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i_m', '--input_directory_music', help='Directory containing music tensors.', required=True)
    parser.add_argument('-i_nm', '--input_directory_nonmusic', help='Directory container non-music tensors.',  required=True)
    args = parser.parse_args()

    all_ids_music = set()
    all_ids_nonmusic = set()

    for filename in os.listdir(args.input_directory_music):
        id = filename.split('.')[0]
        all_ids_music.add(id)

    for filename in os.listdir(args.input_directory_nonmusic):
        id = filename.split('.')[0]
        all_ids_nonmusic.add(id)

    ids_music = list(all_ids_music)
    ids_nonmusic = list(all_ids_nonmusic)

    print(ids_music)
    print(ids_nonmusic)

    music_feature_tensors, music_label_tensors = load_tensors(ids_music, args.input_directory_music, 1)
    nonmusic_feature_tensors, nonmusic_label_tensors = load_tensors(ids_nonmusic, args.input_directory_nonmusic, 0)

    print(music_feature_tensors[0])
    print(np.shape(music_feature_tensors))
    print(np.shape(music_label_tensors))
    print(np.shape(music_feature_tensors))
    print(np.shape(music_label_tensors))

    x_train = np.concatenate([music_feature_tensors, nonmusic_feature_tensors])
    y_train = np.concatenate([music_label_tensors, nonmusic_label_tensors])

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    print(np.shape(x_train))
    print(np.shape(y_train))

    model = Sequential()

    #model.add(LSTM(units=128, dropout=0.30, recurrent_dropout=0.35, return_sequences=True, input_shape=(128,1292)))
    model.add(LSTM(units=128, dropout=0.30, recurrent_dropout=0.25, return_sequences=True, input_shape=(1292,128)))
    model.add(LSTM(units=32, dropout=0.30, recurrent_dropout=0.25, return_sequences=False))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, shuffle=True, batch_size=20, epochs=1000, validation_split=0.3)
