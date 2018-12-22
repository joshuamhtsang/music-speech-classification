# music-speech-classification
Machine learning for speech and music classification

All script should be run from the root directory of this project.

-------------------------------------
--- How to run prepare GTZAN data ---
-------------------------------------

You should have \*.wav files for music and speech in separate directories.  Then,
feed these directory paths as arguments into process_wav_files.py.  For music directory:

$ python process_wav_files.py -i ./datasets/music_speech/music_wav/ -ot tensors_music/  -oimg images_melspec_music

and for speech directory:

$ python process_wav_files.py -i ./datatsets/music_speech/speech_wav/ -ot tensors_nonmusic/  -oimg images_melspec_nonmusic

---------------------------------------
--- How to run the network training ---
---------------------------------------

Run the Docker image:

$ docker run -it -p 8888:8888 -v $PWD:/notebook -e KERAS_BACKEND=tensorflow ermaker/keras-jupyter bash

Once inside the container, run:

$ pip install tensorflow --upgrade
$ pip install tqdm
$ pip install sklearn

The network is defined in lstm_config1.py:

$ python ./ml_scripts/lstm_config1.py -i_m ./datasets/tensors_music/ -i_nm ./datasets/tensors_nonmusic/
