import os.path

from pydub import AudioSegment
import pandas as pd
import numpy as np
import re
import datetime

from preprocessing.preprocessor import Preprocessor
from dataloader.dataset import AudioDataset
from util.helpers import create_directory, get_sample_names_from_paths, get_sample_name_from_path, \
    dataset_name_to_url_part
from util.decorators import cache_to_file_decorator

class AudioCutter(Preprocessor):
    """
    Cut audio files based on known segments to only include parts where patient is speaking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Audio Cutter"
        self.store_segments_to_disk = True
        print(f"Initializing {self.name}")

    def _complete_segmentation(self, segmentation_df):
        """
        Sometimes, the segmentation file only ras rows for the interviewer (INV). The rest should be PAR.
        This method takes the segmentation dataframe and introduces PAR sections accordingly
        """
        # all time boundaries (beginning / ends)
        time_boundaries = np.concatenate((np.array([0]), segmentation_df['begin'], segmentation_df['end']))
        time_boundaries_sorted = np.sort(np.unique(time_boundaries))

        # create rows for each segment
        rows = np.vstack((time_boundaries_sorted[:-1], time_boundaries_sorted[1:])).T
        rows_df = pd.DataFrame(rows, columns=['begin', 'end'])

        # join to existing data
        new_segmentation = segmentation_df.merge(rows_df, on=['begin', 'end'], how="right")

        # fill speaker = PAR where null (not in original df)
        new_segmentation['speaker'] = np.where(~new_segmentation['speaker'].isna(), new_segmentation['speaker'], "PAR")

        return new_segmentation

    def _setup_target_dir(self, target_dir, dataset):
        """
        Write readme file to target folder so it's clear where the data is coming from.
        Also delete potential old files
        """
        create_directory(target_dir)
        with open(os.path.join(target_dir, 'readme.txt'), 'w') as f:
            readme = f"Preprocessed audio from dataset {dataset} using {self.name} on {datetime.datetime.now()}"
            f.write(readme)

    @cache_to_file_decorator()
    def cut(self, audio_file_path, segments_file_path, target_directory):
        # importing file from location by giving its path
        sound = AudioSegment.from_file(audio_file_path, format="wav")

        # segmentation df
        segmentation = pd.read_csv(segments_file_path)
        # complete potentially missing PAR rows in segmentation file
        segmentation = self._complete_segmentation(segmentation)
        # get only PAR segments (from participant, not interviewer)
        segmentation = segmentation.query("speaker == 'PAR'")

        # extract the actual audio segments
        segments = [sound[seg['begin']:seg['end']] for i, seg in segmentation.iterrows()]

        # store segments to disk, for debugging
        if self.store_segments_to_disk:
            create_directory(os.path.join(target_directory, "segments"))
            segments_path = lambda i, file: os.path.join(target_directory, "segments", f"{file}_{i}.wav")
            [seg.export(segments_path(i, os.path.basename(audio_file_path)), format="wav") for i, seg in enumerate(segments)]

        # sum these to get a new entire audio
        new_audio = sum(segments)

        # Saving file in required location
        create_directory(target_directory)
        new_path = os.path.join(target_directory, os.path.basename(audio_file_path))
        new_audio.export(new_path, format="wav")

        return new_path

    def cut_dataset(self, dataset: AudioDataset) -> AudioDataset:
        print(f"Cutting only PAR segments from dataset {dataset.name} using {self.name}")

        target_dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "segmented_audio",
                                  dataset_name_to_url_part(dataset.name))
        self._setup_target_dir(target_dir, dataset)

        dataset_type = None
        if 'ADReSSo' in dataset.name:
            dataset_type = 'ADReSSo'
        elif 'ADReSS' in dataset.name:
            dataset_type = 'ADReSS'
        assert dataset_type is not None, "AudioCutter only works for ADReSS and ADReSSo datasets"

        def segmentation_file_path_ADReSS(file_path):
            """
            For ADReSS dataset. Make sure segmentation files have been written before using
            src/scripts/2023_08_30_segmentation_file_ADReSS.py
            """
            sample_name = get_sample_name_from_path(file_path)
            return os.path.join(self.CONSTANTS.DATA_ADReSS_SEGMENTATION, sample_name + ".csv")

        def target_dir_path_ADReSS(file_path):
            """
            For ADReSS dataset
            Todo: Move all of this matching logic into helpers file at some point...
            """
            # match train/audio/ad/adrso001.wav, test-dist/audio/adrsdt1.wav, etc.
            parts = re.match(r"(.*)(train/|test/)(Full_wave_enhanced_audio/)(cc/|cd/)?(.*)\.wav", file_path)
            if parts is None:
                raise ValueError("Invalid ADReSS file path")
            if parts.group(4) is not None:
                # train split -> get train/audio/ad or train/audio/cn
                return os.path.join(target_dir, parts.group(2), parts.group(4))
            else:
                # test split (no cc / cd section in path)
                return os.path.join(target_dir, parts.group(2))


        if dataset_type == 'ADReSS':
            # ADReSSo dataset
            cut_files = np.array([self.cut(file, segmentation_file_path_ADReSS(file), target_dir_path_ADReSS(file))
                                  for file in dataset.data])

        config_without_preprocessors = {key: dataset.config[key] for key in dataset.config if key != 'preprocessors'}
        new_config = {
            'preprocessors': [*dataset.config['preprocessors'], self.name],
            **config_without_preprocessors
        }
        return AudioDataset(data=cut_files, labels=np.array(dataset.labels),
                            sample_names=get_sample_names_from_paths(cut_files),
                            name=f"{dataset.name} - {self.name}", config=new_config)

    def preprocess_dataset(self, dataset: AudioDataset) -> AudioDataset:
        return self.cut_dataset(dataset)

