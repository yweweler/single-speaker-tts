# LJ Speech Dataset Preparation Example

This is an step by step example on how to prepare the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) 
for training a TTS model.
The preparation procedure follows the steps described in [README.md](README.md):

## Download

First download and unpack the dataset.

```bash
# Download the LJSpeech dataset.
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# Unpack the dataset.
tar -xf LJSpeech-1.1.tar.bz2
rm LJSpeech-1.1.tar.bz2
```

Let's take a look at how the dataset looks like:
```bash
# The folder structure looks like this:
LJSpeech-1.1
├── metadata.csv
├── README
└── wavs
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...

# The audio files contained are of the following format.
file LJSpeech-1.1/wavs/LJ013-0218.wav
>> LJ013-0218.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 22050 Hz
```

The transcriptions contained in the dataset are given in `metadata.csv` like this:
```bash
LJ001-0001|Printing, in the only sense with which ...
LJ001-0002|in being comparatively ...
...
```
Hence, they are of the form `<file-name>|<transcription>/r/n`.
Luckily this project already comes with a custom loader for the LJ Speech dataset to we do not have to write out own.
The loader is defined in [datasets/lj_speech.py](datasets/lj_speech.py).
Based on `metadata.csv` the loader parses the transcriptions and loads the audio files from the `wavs/` folder.


## Calculate And Set The Signal Statistics

Let us now configure the project so that it does use the dataset for training.
First we will set up the dataset [tacotron/params/dataset.py](tacotron/params/dataset.py).
Configure the path to the dataset `dataset_folder` and set the `dataset_loader` to be `LJSpeechDatasetHelper`.
Next, you need to establish an enumerated vocabulary for the dataset and tell the architecture the vocabulary size.

However, as we do not have this information at hand we will have to collect the, first using [tacotron/dataset_statistics.py](tacotron/dataset_statistics.py).
The script will use the dataset loader and the folder path we just configured to collect the missing parameters.

```bash
# Collect the necessary data.
python tacotron/dataset_statistics.py

Dataset: /my-dataset-path/LJSpeech-1.1
Loading dataset ...
Dataset vocabulary:
vocabulary_dict={
    'pad': 0,
    'eos': 1,
    'p': 2,
    'r': 3,
    'i': 4,
    'n': 5,
    't': 6,
    'g': 7,
    ' ': 8,
    'h': 9,
    'e': 10,
    'o': 11,
    'l': 12,
    'y': 13,
    's': 14,
    'w': 15,
    'c': 16,
    'a': 17,
    'd': 18,
    'f': 19,
    'm': 20,
    'x': 21,
    'b': 22,
    'v': 23,
    'u': 24,
    'k': 25,
    'j': 26,
    'z': 27,
    'q': 28,
},
vocabulary_size=29


Collecting decibel statistics for 13100 files ...
mel_mag_ref_db =  6.026512479977281
mel_mag_max_db =  -99.89414986824931
linear_ref_db =  35.65918850818663
linear_mag_max_db =  -100.0
```

Now we can complement `vocabulary_dict` and `vocabulary_size` into the dataset configuration in 
[tacotron/params/dataset.py](tacotron/params/dataset.py).
Additionally the are given a set of decibel values the loader requires from normalizing features.
Make sure to set the variables (`mel_mag_ref_db`, `mel_mag_max_db`, `linear_ref_db`, `linear_mag_max_db`) in 
[datasets/lj_speech.py](datasets/lj_speech.py).


## Configuration Of The Architecture

Next we have to define the architecture parameters in [tacotron/params/model.py](tacotron/params/model.py).
For now we will only set `vocabulary_size=29` and `sampling_rate=22050`, such that the 
architecture does work with the LJ Speech dataset.

Depending on your configuration now would be the right time to start the optional feature 
pre-calculation.


## Configuration Of The Training Process

Finally, we will configure the training process in [tacotron/params/training.py](tacotron/params/training.py).


## Training

Now everything is set up to start training.
```bash
python tacotron/train.py
```