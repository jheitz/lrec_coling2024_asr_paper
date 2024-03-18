import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

from dataloader.dataset import TextDataset, AudioDataset
from util.decorators import cache_to_file_decorator
from preprocessing.audio_transcriber import Transcriber


class Wave2Vec2Transcriber(Transcriber):
    def __init__(self, *args, **kwargs):
        self.name = "wave2vec-large"
        super().__init__(*args, **kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h",
                                                           cache_dir=self.CONSTANTS.CACHE_DIR)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", cache_dir=self.CONSTANTS.CACHE_DIR)

        self.transcriber_config = {}  # any config for the transcriber = different versions
        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this


    @cache_to_file_decorator()
    def transcribe_file(self, file_path: str, version_config: str) -> str:
        print(f"Transcribing file {file_path}", end="... ")

        # load file
        waveform, sample_rate = torchaudio.load(file_path)

        # resample to target sample rate
        waveform, sample_rate = self._resample(waveform, sample_rate)

        # preprocess
        waveform = self._preprocess_waveform(waveform)

        # tokenize & predict
        input_values = self.processor(waveform, return_tensors='pt', padding="longest",
                                      sampling_rate=self.sample_rate).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        print(f"{transcription[:20]}...")

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
