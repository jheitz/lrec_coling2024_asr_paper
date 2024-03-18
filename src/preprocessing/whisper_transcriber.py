import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torchaudio

from dataloader.dataset import TextDataset, AudioDataset
from util.decorators import cache_to_file_decorator
from preprocessing.audio_transcriber import Transcriber


class WhisperTranscriber(Transcriber):

    def __init__(self, *args, **kwargs):
        self.name = "whisper-large"
        super().__init__(*args, **kwargs)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            chunk_length_s=30,
            device=device,
        )

        self.transcriber_config = {}  # any config for the transcriber = different versions
        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this


    @cache_to_file_decorator()
    def transcribe_file(self, file_path: str, version_config: str) -> str:
        print(f"Transcribing file {file_path}")

        # load file
        waveform, sample_rate = torchaudio.load(file_path)

        # resample to target sample rate
        waveform, sample_rate = self._resample(waveform, sample_rate)

        # preprocess
        waveform = self._preprocess_waveform(waveform)

        transcription = self.pipe(np.array(waveform), batch_size=4)['text']

        self._save_transcription(file_path, transcription)

        return transcription

    def transcribe_dataset(self, dataset: AudioDataset) -> TextDataset:
        # create new TextDataset with transcribed files
        super().transcribe_dataset(dataset)
        self._initialize_transcription(dataset)

        transcribed_data = np.array([self.transcribe_file(file, version_config=self._version_config)
                                     for file in dataset.data])

        config_without_preprocessors = {key: dataset.config[key] for key in dataset.config if key != 'preprocessors'}
        new_config = {
            'preprocessors': [*dataset.config['preprocessors'], self.name],
            **config_without_preprocessors
        }
        return TextDataset(data=transcribed_data, labels=np.array(dataset.labels),
                           sample_names=np.array(dataset.sample_names),
                           name=f"{dataset.name} - {self.name} transcribed", config=new_config)
