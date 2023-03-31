# transcribe-audio-files

This repository provides a Python script to transcribe a collection of waveform audio files using the OpenAI Whisper model. The script processes audio files in parallel, splits transcriptions into sentences, and exports the results to the specified output directory. It also takes advantage of whisper-timestamped (https://github.com/linto-ai/whisper-timestamped) for processing.

## Features

- Transcribe audio files using OpenAI Whisper ASR model.
- Split transcriptions into sentences using NLTK Punkt tokenizer.
- Process audio files in parallel using multiple worker processes.
- Export transcribed sentences to text files and sliced audio files.
- Checkpoint the list of processed audio files at regular intervals.

## Requirements

- Python 3.6 or higher
- whisper-timestamped
- pydub
- NLTK
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Team-Potion/transcribe-audio-files
   cd transcribe-audio-files
   ```

### For development installation:

2. Install the required packages:
   ```
   pip install -r requirements.dev.txt
   ```

3. Install the package in editable mode:
   ```
   pip install -e .
   ```

### For usage installation:

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install the package:
   ```
   pip install .
   ```

## Usage

Transcribe a collection of waveform audio files using the following command:

```
transcribe-audio-files --input_path <input_dir_path> --output_path <output_dir_path> [--model_name <model_name>] [--language <language>] [--confidence_threshold <confidence_threshold>] [--device <device>] [--num_workers <num_workers>] [--checkpoint_interval <checkpoint_interval>] [--no_split_patterns <no_split_patterns>] [--verbose]
```

Arguments:

- `--input_path` (str): The path to the directory containing the collection of waveform audio files to be transcribed.
- `--output_path` (str): The path to the output directory.
- `--model_name` (str, optional): The name of the OpenAI Whisper model to use. Default is "large-v2".
- `--language` (str, optional): The language for transcribing the audio files. Default is "English".
- `--confidence_threshold` (float, optional): The minimum confidence threshold for considering transcribed words. Default is 0.45.
- `--device` (str, optional): The device to use for processing. Default is "cuda".
- `--num_workers` (int, optional): The number of worker processes to use for parallel processing. Default is 2.
- `--checkpoint_interval` (int, optional): The number of audio files to process before saving the processed files list to disk. Default is 10000.
- `--no_split_patterns` (str or list of str, optional): The naming convention pattern(s) to avoid splitting of audio files. Default is None.
- `--verbose` (bool, default = False, optional): Increase the log level to INFO; default is WARNING.

## Performance

Running on an Ubuntu 22.04 LTS workstation with an Nvidia GPU (GeForce RTX 3090) using the default settings, 200 waveform audio files (90% are comprised of only a single sentence with the remaining 10% averaging 10 sentences per audio file) are transcribed in about 2 minutes.

Running on an AWS-hosted Ubuntu 22.04 LTS instance (type: g5.2xlarge) using the default settings, 140,000 waveform audio files (90% are comprised of only a single sentence with the remaining 10% averaging 10 sentences per audio file) are transcribed in about 16 hours.

## Acknowlegment

- [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped): Multilingual Automatic Speech Recognition with word-level timestamps and confidence (License: AGPL-3.0).
- [whisper](https://github.com/openai/whisper): Robust Speech Recognition via Large-Scale Weak Supervision (License: MIT).
- [Pydub](https://github.com/jiaaro/pydub): Manipulate audio with a simple and easy high level interface (License: MIT)
