#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

from typing import List, Dict, Set, Tuple, Optional, Any

from multiprocessing import Pool, Manager, Value, Lock
from functools import partial

import pickle
from tqdm import tqdm

import whisper_timestamped as whisper
from pydub import AudioSegment

# NLTK imports for splitting text into sentences
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# enable logging
from .logging_utils import get_logger
logger = get_logger(__name__)


def parse_cmdline_args() -> argparse.Namespace:
    """
    parse_cmdline_args - function to parse command line arguments for transcribing a collection of waveform audio files

    Arguments:
        --input_path (str): The path to the directory containing the collection of waveform audio files to be transcribed.
        --output_path (str): The path to the output directory.
        --model_name (str): The name of the OpenAI Whisper model to use. Default is "large-v2".
        --language (str): The language for transcribing the audio files. Default is "English".
        --confidence_threshold (float): The minimum confidence threshold for considering transcribed words. Default is 0.45.
        --device (str): The device to use for processing. Default is "cuda".
        --num_workers (int): The number of worker processes to use for parallel processing. Default is 2.
    
    Returns:
        argparse.Namespace: A namespace containing the parsed arguments.
    """

    # create argument parser object
    parser = argparse.ArgumentParser(
        description = "Transcribe a Collection of Waveform Audio Files using whisper-timestamped (https://github.com/linto-ai/whisper-timestamped).")
    
    # add command line arguments
    parser.add_argument("--input_path",           type = str, required = True,
        help = "The path to the directory containing the collection of waveform audio files to be transcribed.")
    parser.add_argument("--output_path",          type = str, required = True,
        help = "The path to the output directory.")
    parser.add_argument("--model_name",           type = str, default = "large-v2",
        help = "The name of the OpenAI Whisper model to use.")
    parser.add_argument("--language",             type = str, default = "English",
        help = "The target language for transcribing the audio files.")
    parser.add_argument("--confidence_threshold", type = float, default = 0.45,
        help = "The minimum confidence threshold for considering transcribed words / sentences.")
    parser.add_argument("--device",               type = str, default = "cuda",
        help = "The device to use for processing.")
    parser.add_argument("--num_workers",          type = int, default = 2,
        help = "The number of worker processes to use for parallel processing.")

    # return the parsed arguments
    return parser.parse_args()


def download_tokenizer(language: str) -> None:
    """
    Download the Punkt tokenizer for the specified language if it is not already available.

    Args:
        language (str): The target language for which the tokenizer is needed.

    Returns:
        None
    """

    try:
        nltk.data.find(f"tokenizers/punkt/{language.lower()}.pickle")
    except LookupError:
        nltk.download("punkt", quiet = True)


def load_tokenizer(language: str) -> Optional[PunktSentenceTokenizer]:
    """
    load_tokenizer - function to load the Punkt tokenizer for the given language

    Arguments:
        language (str): The target language for which to load the tokenizer.
    
    Returns:
        Optional[PunktSentenceTokenizer]: A PunktSentenceTokenizer object if the tokenizer is found for the given language, otherwise None.
    """

    try:
        tokenizer = nltk.data.load(f"tokenizers/punkt/{language.lower()}.pickle")
    except LookupError:
        print(f"Error: No Punkt tokenizer found for '{language}'.")
        return None
    return tokenizer


def transcribe_audio(audio_file: str, language: str) -> Dict:
    """
    transcribe_audio - function to transcribe an audio file using the Whisper ASR system

    Arguments:
        audio_file (str): The path to the audio file to be transcribed.
        language (str): The target language for transcribing the audio file.

    Returns:
        Dict: A dictionary containing the transcription result with start and end times, text, and confidence scores for each word.
    """

    global model

    # load the audio file to be transcribed
    try:
        audio = whisper.load_audio(audio_file)
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        return {}

    # transcribe the audio file using the loaded model
    try:
        transcription_rslt = whisper.transcribe(model, audio, language = language, verbose = None)
    except Exception as e:
        logger.error(f"Error transcribing audio file: {e}")
        return {}

    return transcription_rslt


def transcription_to_sentences(transcription_rslt: Dict, language: str) -> Tuple[List[Dict], List[float]]:
    """
    transcription_to_sentences - function to convert the transcription result into sentences, align confidence scores, and determine cut points

    Arguments:
        transcription_rslt (Dict): A dictionary containing the transcription result with start and end times, text, and confidence scores for each word.
        language (str): The target language for transcribing the audio file.

    Returns:
        Tuple[List[Dict], List[float]]: A tuple containing two lists:
            1. A list of dictionaries, each containing the start and end times, text, and confidence scores for each sentence.
            2. A list of cut points (in seconds) between sentences.
    """

    words = []

    # iterate through segments in the transcription result
    for segment in transcription_rslt["segments"]:
        # iterate through words within each segment
        for word in segment["words"]:
            words.append({"start": word["start"], "end": word["end"], "text": word["text"].strip(), "confidence": word["confidence"]})

    # join words into a single string
    full_text = ' '.join(word["text"] for word in words)

    # load the Punkt tokenizer for the specified language
    tokenizer = load_tokenizer(language)

    # ensure the tokenizer has been loaded successfully
    if tokenizer is None:
        logger.error(f"Error: No tokenizer loaded.")
        return [], []

    # split the text into sentences
    sentences = tokenizer.tokenize(full_text)

    # align the words with the sentences and their corresponding confidence scores
    aligned_sentences = []
    sentence_cuts = []
    word_idx = 0

    # sentence cut points need to include the start of the first sentence
    sentence_cuts.append(words[word_idx]["start"] / 2)

    # iterate over all sentences
    for sentence in sentences:
        sentence_start = words[word_idx]["start"]
        sentence_words = []
        sentence_confidence_scores = []

        while word_idx < len(words) and sentence.startswith(words[word_idx]["text"]):
            sentence_words.append(words[word_idx]["text"])
            sentence_confidence_scores.append(words[word_idx]["confidence"])
            sentence_start = words[word_idx]["start"]
            sentence = sentence[len(words[word_idx]["text"]):].lstrip()
            word_idx += 1

        # if the sentence is not empty, it means the last word was not added, so add it
        if sentence:
            sentence_words.append(words[word_idx - 1]["text"])
            sentence_confidence_scores.append(words[word_idx - 1]["confidence"])

        # calculate the end of the sentence
        sentence_end = words[word_idx - 1]["end"]

        aligned_sentences.append({"start": sentence_start, "end": sentence_end, "text": ' '.join(sentence_words), "confidence_score": sum(sentence_confidence_scores) / len(sentence_confidence_scores)})

        # calculate cut points
        if word_idx < len(words):
            sentence_cuts.append((sentence_end + words[word_idx]["start"]) / 2)

    # sentence cut points need to include the end of the last sentence
    sentence_cuts.append(sentence_end)

    return aligned_sentences, sentence_cuts


def copy_file(src: str, dst: str) -> None:
    """
    Copy a file from the source path to the destination path.

    Args:
        src (str): The source file path.
        dst (str): The destination file path.

    Returns:
        None
    """

    try:
        shutil.copy(src, dst)
    except Exception as e:
        logger.error(f"Error copying file: {e}")


def load_processed_files(processed_files_path: str) -> Set[str]:
    """
    Load the set of processed files from the specified path.

    Args:
        processed_files_path (str): The path to the file containing the set of processed files.

    Returns:
        Set[str]: A set of processed file paths, or an empty set if the specified file does not exist.
    """

    if os.path.exists(processed_files_path):
        try:
            with open(processed_files_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
            return set()
    else:
        return set()


def save_processed_files(processed_files: Set[str], processed_files_path: str) -> None:
    """
    Save the set of processed files to the specified path.

    Args:
        processed_files (Set[str]): The set of processed file paths.
        processed_files_path (str): The path to the file where the set of processed files should be saved.

    Returns:
        None
    """

    try:
        with open(processed_files_path, "wb") as f:
            pickle.dump(processed_files, f)
    except Exception as e:
        logger.error(f"Error saving processed files: {e}")


def process_audio_file(input_path: str, output_path: str, language: str, confidence_threshold: float) -> None:
    """
    process_audio_file - function to process a single audio file by transcribing it, splitting sentences, and exporting the results

    Arguments:
        input_path (str): The path to the input audio file to be processed.
        output_path (str): The path to the output directory where the results will be saved.
        language (str): The target language for transcribing the audio file.
        confidence_threshold (float): The minimum confidence threshold for considering transcribed words / sentences.

    Returns:
        None
    """

    # transcribe audio
    transcription = transcribe_audio(input_path, language)

    # ensure a transcription has been returned
    if transcription == {}:
        logger.error(f"Error: No transcription result for {input_path}.")
        return

    # ensure a segments section is in the transcription
    if not transcription["segments"]:
        logger.error(f"Error: Transcription result for {input_path} is missing the 'segments' key.")
        return

    # split sentences, align confidence scores, and determine cut points
    aligned_sentences, sentence_cuts = transcription_to_sentences(transcription, language)

    # test if there are more than one sentence in the transcription result
    if (len(aligned_sentences) > 1):

        # remove the original extension from output_path
        output_path_wo_ext, _ = os.path.splitext(output_path)

        # slice the input_path file (by sentences)
        audio = AudioSegment.from_wav(input_path)
        for i, (start, end) in enumerate(zip(sentence_cuts[:-1], sentence_cuts[1:]), start = 1):
            # determine if the sentence's confidence_score is above the threshold
            if aligned_sentences[i - 1]["confidence_score"] < confidence_threshold:
                logger.info(f"Dropping sentence {i:03d} from ({input_path}) due to low confidence score ({aligned_sentences[i - 1]['confidence_score']}).")
                continue
 
            # slice the audio files by sentence
            try:
                sliced_audio = audio[max(0, int(start * 1000)):int(end * 1000)]
                sliced_audio.export(f"{output_path_wo_ext}_sentence_{i:03d}.wav", format = "wav")
            except Exception as e:
                logger.error(f"Error slicing audio: {e}")

        # save transcriptions for each sliced audio segment
        for i, sentence in enumerate(aligned_sentences, start = 1):
            # determine if the sentence's confidence_score is above the threshold
            if aligned_sentences[i - 1]["confidence_score"] < confidence_threshold:
                logger.info(f"Dropping sentence {i:03d} from ({input_path}) due to low confidence score ({aligned_sentences[i - 1]['confidence_score']}).")
                continue

            # save transcriptions by sentence
            try:
                with open(f"{output_path_wo_ext}_sentence_{i:03d}.txt", "w") as f:
                    f.write(sentence["text"].strip())
            except IOError as e:
                logger.error(f"Error writing to file {output_path_wo_ext}_sentence_{i:03d}.txt: {e}")
    else:
        # determine if the sentence's confidence_score is above the threshold
        if aligned_sentences[0]["confidence_score"] < confidence_threshold:
            logger.info(f"Dropping recording ({input_path}) due to low confidence score ({aligned_sentences[0]['confidence_score']}).")
            return

        # copy waveform to output folder
        copy_file(input_path, output_path)

        # write transcription to file
        try:
            with open(f"{output_path}.txt", "w") as f:
                f.write(aligned_sentences[0]["text"].strip())
        except IOError as e:
            logger.error(f"Error writing to file {output_path}.txt: {e}")

    return


def process_audio_file_wrapper(args: Tuple[str, str, str, float], shared_counter: Value, counter_lock: Lock, processed_files: dict, processed_files_path: str, save_interval: int) -> Any:
    """
    process_audio_file_wrapper - wrapper function to pass multiple arguments to the process_audio_file function for use with multiprocessing

    Arguments:
        args (Tuple[str, str, str, float]): A tuple containing the following arguments:
            1. input_path (str): The path to the input audio file to be processed.
            2. output_path (str): The path to the output directory where the results will be saved.
            3. language (str): The target language for transcribing the audio file.
            4. confidence_threshold (float): The minimum confidence threshold for considering transcribed words / sentences.
        shared_counter (Value): A shared counter for tracking the number of processed audio files.
        counter_lock (Lock): A lock for synchronizing access to the shared counter.
        processed_files (dict): A shared dictionary containing the paths of the processed audio files as keys.
        processed_files_path (str): The path to the file containing the serialized set of processed audio files.
        save_interval (int): The number of processed files to save before writing to disk.

    Returns:
        Any: The return value of the process_audio_file function (None in this case).
    """

    # call the process_audio_file function with the provided arguments
    result = process_audio_file(*args)

    # safely increment the shared counter using the counter_lock
    with counter_lock:
        input_file_path = args[0]   # extract the input file path from the args tuple
        processed_files[input_file_path] = None   # add the input file path to the processed_files dictionary
        shared_counter.value += 1

        # save the processed files when the counter reaches the save_interval
        if shared_counter.value % save_interval == 0:
            save_processed_files(set(processed_files.keys()), processed_files_path)

    return result


def init_worker(model_name: str, device: str) -> None:
    """
    init_worker - function to initialize a worker process by loading the ASR model into memory for use in multiprocessing

    Arguments:
        model_name (str): The name of the ASR model to be loaded.
        device (str): The device to run the ASR model on, e.g., 'cuda' or 'cpu'.

    Returns:
        None
    """

    global model
    model = whisper.load_model(model_name, device = device)


def managed_set(manager: Manager) -> dict:
    """
    Create a managed set using a Manager object.

    Arguments:
        manager (Manager): A Manager object.

    Returns:
        dict: A dictionary with set-like operations.
    """
    return manager.dict(enumerate(set()))



def process_audio_files(input_path: str, output_path: str, model_name: str, language: str, confidence_threshold: float, device: str, num_workers: int, save_interval: int = 10000) -> None:
    """
    process_audio_files - function to process a collection of audio files by transcribing, splitting sentences, and exporting the results using parallel processing

    Arguments:
        input_path (str): The path to the directory containing the collection of waveform audio files to be processed.
        output_path (str): The path to the output directory where the results will be saved.
        model_name (str): The name of the OpenAI Whisper model to use.
        language (str): The target language for transcribing the audio files.
        confidence_threshold (float): The minimum confidence threshold for considering transcribed words / sentences.
        device (str): The device to use for processing, e.g., 'cuda' or 'cpu'.
        num_workers (int): The number of worker processes to use for parallel processing.
        save_interval (int): The number of processed files to save before writing to disk.

    Returns:
        None
    """

    # get a list of all waveform files in the input directory and sort them by modification time
    wav_files = [f for f in os.listdir(input_path) if f.lower().endswith(".wav")]
    wav_files.sort(key = lambda x: os.path.getmtime(os.path.join(input_path, x)))

    # load previously processed files
    processed_files_path = os.path.join(output_path, "processed_files.pkl")
    processed_files = load_processed_files(processed_files_path)

    # create a list of tasks where each task represents the processing of a single audio file
    tasks = []
    for filename in wav_files:
        input_file_path = os.path.join(input_path, filename)
        output_file_path = os.path.join(output_path, filename)

        # skip files that have already been processed
        if input_file_path in processed_files:
            continue
        else:
            processed_files.add(input_file_path)

        tasks.append((input_file_path, output_file_path, language, confidence_threshold))

    # create a multiprocessing manager
    manager = Manager()

    # create a shared counter and a lock for synchronization
    shared_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()

    # create a shared dictionary for processed files
    processed_files_dict = manager.dict()
    for file in load_processed_files(processed_files_path):
        processed_files_dict[file] = None

    # create a partial function with the shared counter, lock, and other required arguments
    process_audio_file_wrapper_with_shared_data = partial(process_audio_file_wrapper, shared_counter = shared_counter, counter_lock = counter_lock, processed_files = processed_files_dict, processed_files_path=processed_files_path, save_interval = save_interval)

    # create a multiprocessing pool with the specified number of workers
    with Pool(num_workers, initializer = init_worker, initargs = (model_name, device)) as pool:
        # use the partial function in the imap call
        results = list(tqdm(pool.imap(process_audio_file_wrapper_with_shared_data, tasks), total = len(tasks)))

    # save the processed files to disk using the updated managed dictionary
    save_processed_files(set(processed_files_dict.keys()), processed_files_path)


def main() -> None:
    """
    main - the main function to transcribe a collection of waveform audio files using whisper-timestamped (https://github.com/linto-ai/whisper-timestamped).

    Usage: transcribe-audio-files --input_path <input_dir_path> --output_path <output_dir_path> [--model_name <model_name>] [--language <language>] [--confidence_threshold <confidence_threshold>] [--device <device>] [--num_workers <num_workers>]

    Arguments:
        --input_path (str): The path to the directory containing the collection of waveform audio files to be transcribed.
        --output_path (str): The path to the output directory.
        --model_name (str): The name of the OpenAI Whisper model to use. Default is "large-v2".
        --language (str): The language for transcribing the audio files. Default is "English".
        --confidence_threshold (float): The minimum confidence threshold for considering transcribed words. Default is 0.45.
        --device (str): The device to use for processing. Default is "cuda".
        --num_workers (int): The number of worker processes to use for parallel processing. Default is 2.
    
    Returns:
        None
    """

    # parse command line arguments
    args = parse_cmdline_args()

    # ensure the output path exists
    os.makedirs(args.output_path, exist_ok = True)

    # download NLTK Punkt tokenizer in the specified language
    download_tokenizer(args.language)

    # call the main function
    process_audio_files(args.input_path, args.output_path, args.model_name, args.language, args.confidence_threshold, args.device, args.num_workers)
