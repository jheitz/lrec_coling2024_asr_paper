from abc import abstractmethod
import os, re
import numpy as np
import pandas as pd
# import pylangacq
import time

from dataloader.dataset import AudioDataset, TextDataset
from dataloader.cv_shuffler import ADReSSCrossValidationShuffler
from config.constants import Constants
from util.helpers import create_directory, get_sample_names_from_paths, hash_from_dict, dataset_name_to_url_part
from dataloader.chat_parser import ChatTranscriptParser


class DataLoader:
    def __init__(self, name, debug=False, local=None, constants=None, config=None):
        self.debug = debug
        self.name = name
        self.preprocessors = []
        if constants is None:
            self.CONSTANTS = Constants(local=local)
        else:
            self.CONSTANTS = constants
        self.config = config
        print(f"Initializing dataloader {self.name}")

    @abstractmethod
    def _load_train(self):
        pass

    @abstractmethod
    def _load_test(self):
        pass

    def load_data(self):
        print(f"Loading data using dataloader {self.name}")
        train = self._load_train()
        test = self._load_test()
        return train, test


class ADReSSDataLoader(DataLoader):

    def __init__(self, name="ADReSS audio", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.preprocessors = []
        self.dir_train_ad = self.CONSTANTS.DATA_ADReSS_TRAIN_AD
        self.dir_train_cc = self.CONSTANTS.DATA_ADReSS_TRAIN_CONTROL
        self.dir_test = self.CONSTANTS.DATA_ADReSS_TEST

    def _load_train(self):
        paths = []
        labels = []
        for dir_path in [self.dir_train_ad, self.dir_train_cc]:
            label = 1 if os.path.basename(dir_path) == 'cd' else 0  # cc = control or cd = dementia
            for i, file_name in enumerate(os.listdir(dir_path)):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(dir_path, file_name)
                    paths.append(file_path)
                    labels.append(label)
                if self.debug and i >= 1:
                    break

        dataset = AudioDataset(data=np.array(paths), labels=np.array(labels),
                               sample_names=get_sample_names_from_paths(np.array(paths)), name=f"{self.name} (train)",
                               config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                       'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS)})
        return dataset

    def _load_test(self):
        paths = []
        for i, file_name in enumerate(os.listdir(self.dir_test)):
            if file_name.endswith('.wav'):
                file_path = os.path.join(self.dir_test, file_name)
                file_id = re.sub(r'\.wav', '', os.path.basename(file_path))
                paths.append((file_id, file_path))
            if self.debug and i > 1:
                break

        paths = pd.DataFrame(paths, columns=['ID', 'path'])

        assert os.path.exists(self.CONSTANTS.DATA_ADReSS_TEST_METADATA), "Test label / metadata file not available"

        metadata = pd.read_csv(self.CONSTANTS.DATA_ADReSS_TEST_METADATA, delimiter=';')
        metadata.columns = [c.strip() for c in metadata.columns]
        metadata['ID'] = metadata['ID'].str.strip()

        data = metadata.merge(paths, on="ID", how="inner")
        # labels = np.squeeze(metadata['Label'])

        dataset = AudioDataset(data=np.array(data['path']), labels=np.array(data['Label']),
                               sample_names=get_sample_names_from_paths(np.array(data['path'])), name=f"{self.name} (test)",
                               config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                       'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS)})
        return dataset


class ADReSSTranscriptDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        # for this dataloader, we consider different configurations to load / preprocess CHAT format differently
        # in different situations, we sometimes want the configurations to be explicitly passed in the constructor,
        # while other times we want to use config given in the run yaml file

        config_settings = ['only_PAR', 'keep_pauses', 'keep_terminators', 'keep_unintelligable_speech']

        name = "ADReSS manual transcripts"
        kwargs_without_explicit_config = {k: kwargs[k] for k in kwargs if k not in config_settings}
        super().__init__(name, *args, **kwargs_without_explicit_config)

        # back to the config: first, let's load the explicit parameters from kwargs
        transcript_config_dict = {k: kwargs[k] for k in kwargs if k in config_settings}

        # then, let's load the remaining ones from self.config.config_data
        if hasattr(self.config, 'config_data'):
            new_vals = {k: getattr(self.config.config_data, k) for k in vars(self.config.config_data)
                        if k in config_settings and k not in transcript_config_dict}
            transcript_config_dict = {**transcript_config_dict, **new_vals}

        # lastly, let's get default values for what is not defined yet
        self.transcript_config = {
            'only_PAR': transcript_config_dict.get('only_PAR', True), # only keep participant (PAR) parts if only_PAR==True, otherwise also include interviewer INV
            'keep_pauses': transcript_config_dict.get('keep_pauses', False),
            'keep_terminators': transcript_config_dict.get('keep_terminators', False),
            'keep_unintelligable_speech': transcript_config_dict.get('keep_unintelligable_speech', False)
        }
        self.transcript_config_hash = hash_from_dict(self.transcript_config, 6)

        self.method = None  # Choose method = 'pylangacq' for word extraction based on pylangacq library
        print("ADReSSTranscriptDataLoader chat_config:", self.transcript_config)
        self.preprocessors = []
        self.chat_transcript_parser = ChatTranscriptParser(config=self.transcript_config)

    def _save_transcription(self, group, split, file_path, transcription):
        """ Save preprocessed transcript to file so it can be looked at """
        metadata = {"file": file_path, "group": group, "split": split}

        try:
            # from /path/to/S123.wav -> extract S123
            identifier = re.match(r".*/([a-zA-Z0-9_-]+.)\.(cha)", file_path).group(1)
            metadata["identifier"] = identifier
        except:
            print("Failed identifier of transcription, cannot save")
            return False

        def save_with_metadata():
            # save including some metadata
            content = "\n".join(f"{key}: {value}" for key, value in metadata.items())
            content += "\n\n"
            content += transcription
            transcriptions_base_dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "transcriptions", "manual",
                                                   dataset_name_to_url_part(self.name),
                                                   self.transcript_config_hash)
            transcriptions_dir = os.path.join(transcriptions_base_dir, split if split != "NA" else "", group)
            create_directory(transcriptions_dir)

            config_file = os.path.join(transcriptions_base_dir, "config.txt")
            with open(config_file, 'w') as f:
                f.write(str({**self.transcript_config, 'runtime': time.strftime("%Y-%m-%d %H:%M")}))

            file_to_store = os.path.join(transcriptions_dir, identifier + ".txt")
            with open(file_to_store, 'w') as f:
                return f.write(content)

        def save_raw():
            # save raw content, for further analysis
            transcriptions_base_dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "transcriptions", "manual",
                                                   "_raw_" + dataset_name_to_url_part(self.name),
                                                   self.transcript_config_hash)
            transcriptions_dir = os.path.join(transcriptions_base_dir, split if split != "NA" else "", group)
            create_directory(transcriptions_dir)

            file_to_store = os.path.join(transcriptions_dir, identifier + ".txt")
            with open(file_to_store, 'w') as f:
                return f.write(transcription)


        save_raw()
        save_with_metadata()


    def _load_train(self):
        transcripts = []
        paths = []
        labels = []
        disfluency_metrics = []
        for dir_path in [self.CONSTANTS.DATA_ADReSS_TRAIN_TRANSCRIPTS_AD,
                         self.CONSTANTS.DATA_ADReSS_TRAIN_TRANSCRIPTS_CONTROL]:
            label = 1 if os.path.basename(dir_path) == 'cd' else 0  # cc = control or cd = dementia
            for i, file_name in enumerate(os.listdir(dir_path)):
                if file_name.endswith('.cha'):
                    file_path = os.path.join(dir_path, file_name)
                    if self.method == 'pylangacq':
                        transcription = self._get_transcript_using_pylangacq(file_path)
                        self._save_transcription(os.path.basename(dir_path), "train", file_path, transcription)
                        transcripts.append(transcription)
                    else:
                        with open(file_path, 'r') as file:
                            transcript = file.read()
                            preprocessed = self.chat_transcript_parser.preprocess_transcript(transcript)
                            disfluency_metrics.append(self.chat_transcript_parser.extract_disfluency_metrics(transcript))
                            self._save_transcription(os.path.basename(dir_path), "train", file_path, preprocessed)
                        transcripts.append(preprocessed)
                    labels.append(label)
                    paths.append(file_path)
                if self.debug and i >= 1:
                    break

        dataset = TextDataset(data=np.array(transcripts), labels=np.array(labels),
                              sample_names=get_sample_names_from_paths(np.array(paths)), name=f"{self.name} (train)",
                              paths=np.array(paths), config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                                   'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS),
                                                   'transcript_config': self.transcript_config,
                                                   'transcript_config_hash': self.transcript_config_hash},
                              # Add disfluency metric as own piece of data, to be used by linguistic feature extractor
                              disfluency_metrics=pd.DataFrame(disfluency_metrics))
        return dataset

    def _load_test(self):
        info = []
        for i, file_name in enumerate(os.listdir(self.CONSTANTS.DATA_ADReSS_TEST_TRANSCRIPTS)):
            if file_name.endswith('.cha'):
                file_path = os.path.join(self.CONSTANTS.DATA_ADReSS_TEST_TRANSCRIPTS, file_name)
                file_id = re.sub(r'\.cha', '', os.path.basename(file_path))
                if self.method == 'pylangacq':
                    parsed_cha = pylangacq.read_chat(file_path)
                    transcription = " ".join(parsed_cha.words(participants='PAR' if self.transcript_config['only_PAR'] else None))
                    self._save_transcription("NA", "test", file_path, transcription)
                    info.append((file_id, transcription, file_path, []))
                else:
                    with open(file_path, 'r') as file:
                        transcript = file.read()
                        preprocessed = self.chat_transcript_parser.preprocess_transcript(transcript)
                        disfluency_metrics = self.chat_transcript_parser.extract_disfluency_metrics(transcript)
                        self._save_transcription("NA", "test", file_path, preprocessed)
                    info.append((file_id, preprocessed, file_path, disfluency_metrics))
            if self.debug and i > 1:
                break

        transcript = pd.DataFrame(info, columns=['ID', 'transcript', 'path', 'disfluency_metrics'])

        assert os.path.exists(self.CONSTANTS.DATA_ADReSS_TEST_METADATA), "Test label / metadata file not available"

        metadata = pd.read_csv(self.CONSTANTS.DATA_ADReSS_TEST_METADATA, delimiter=';')
        metadata.columns = [c.strip() for c in metadata.columns]
        metadata['ID'] = metadata['ID'].str.strip()

        data = metadata.merge(transcript, on="ID", how="inner")
        # labels = np.squeeze(metadata['Label'])

        dataset = TextDataset(data=np.array(data['transcript']), labels=np.array(data['Label']),
                              sample_names=get_sample_names_from_paths(np.array(data['path'])), name=f"{self.name} (test)",
                              paths=np.array(data['path']),
                              config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                      'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS),
                                      'transcript_config': self.transcript_config,
                                      'transcript_config_hash': self.transcript_config_hash},
                              # Add disfluency metric as own piece of data, to be used by linguistic feature extractor
                              disfluency_metrics=data['disfluency_metrics'].apply(pd.Series))
        return dataset
