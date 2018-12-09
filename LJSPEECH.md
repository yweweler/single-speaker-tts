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


## Create Listing Files

Luckily this project already comes with a custom listing generator for the LJ Speech dataset so we
 do not have to write out own.
The generator is defined in [datasets/preparation/ljspeech.py](datasets/preparation/ljspeech.py).
Based on `metadata.csv` the generator parses and pre-processes the transcriptions to generate and
 write both the `train.csv` and `eval.csv` listing files.


## Generate the Dataset Definition File

Before the dataset can be used a definition file has to be created.
When both `train.csv` and `eval.csv` exist the dataset definition file can be generated like in 
this:
```python
from datasets.dataset import Dataset

dataset = Dataset('/tmp/LJSpeech-1.1/dataset.json')
dataset.set_dataset_folder('/tmp/LJSpeech-1.1/')
dataset.set_audio_folder('wavs')
dataset.set_train_listing_file('train.csv')
dataset.set_eval_listing_file('eval.csv')
dataset.load_listings(stale=True)
dataset.generate_vocabulary()

# Calculates the signal statistics over the entire dataset (may take a while).
dataset.generate_normalization(n_threads=4)
dataset.save()
```

Let us now configure the project so that it does use the dataset for training.
First we will set up the dataset [tacotron/params/dataset.py](tacotron/params/dataset.py).
Just configure the path to the dataset definition file `dataset_file`.


## Configuration Of The Architecture

Next we have to define the architecture parameters in [tacotron/params/model.py](tacotron/params/model.py).
For now we will only set `vocabulary_size=39` and `sampling_rate=22050`, such that the 
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