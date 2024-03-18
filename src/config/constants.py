import os, sys


class Constants:
    """
    A class of project-wise CONSTANTS.
    Directory paths can depend on being run locally or not (which is given by the --local runtime parameter)
    This local flag can also be set manually (Constants(True) or Constants(False)), or, in any case,
    the local / remote version can be accessed using Constants().LOCAL.CONSTANT_NAME / Constants().REMOTE.CONSTANT_NAME
    """
    def __init__(self, local=None, recursion=True):
        git_dir_remote = "/home/ubuntu/git/dementia"
        git_dir_local = "/Users/jheitz/git/dementia"

        if local is not None:
            self.local = local
        else:
            if "--local" in sys.argv:
                self.local = True
            elif os.path.exists(git_dir_local):
                self.local = True
            else:
                self.local = False

        # Git branch where code snapshots are committed and pushed to by src/run/run.py
        self.EXPERIMENT_BRANCH = 'experiments'

        # path to directory of project git
        self.GIT_DIR = git_dir_remote
        if self.local:
            self.GIT_DIR = git_dir_local

        self.CACHE_DIR = os.path.join(self.GIT_DIR, "cache")
        self.RESOURCES_DIR = os.path.join(self.GIT_DIR, "resources")

        # ADReSS data set
        self.DATA_ADReSS_ROOT = "/home/ubuntu/methlab/Students/Jonathan/data/dementiabank_extracted/0extra/ADReSS-IS2020-data"
        if self.local:
            self.DATA_ADReSS_ROOT = "/Users/jheitz/phd/data/dementiabank_extracted/0extra/ADReSS-IS2020-data"

        self.DATA_ADReSS_TRAIN_CONTROL = os.path.join(self.DATA_ADReSS_ROOT, "train/Full_wave_enhanced_audio/cc")
        self.DATA_ADReSS_TRAIN_AD = os.path.join(self.DATA_ADReSS_ROOT, "train/Full_wave_enhanced_audio/cd")
        self.DATA_ADReSS_TEST = os.path.join(self.DATA_ADReSS_ROOT, "test/Full_wave_enhanced_audio")
        self.DATA_ADReSS_TEST_METADATA = os.path.join(self.DATA_ADReSS_ROOT, "test", "meta_data_with_labels.csv")

        self.DATA_ADReSS_TRAIN_TRANSCRIPTS_CONTROL = os.path.join(self.DATA_ADReSS_ROOT, "train/transcription/cc")
        self.DATA_ADReSS_TRAIN_TRANSCRIPTS_AD = os.path.join(self.DATA_ADReSS_ROOT, "train/transcription/cd")
        self.DATA_ADReSS_TEST_TRANSCRIPTS = os.path.join(self.DATA_ADReSS_ROOT, "test/transcription")

        self.RESULTS_ROOT = "/home/ubuntu/methlab/Students/Jonathan/results"
        self.RESULTS_ROOT_REMOTE = self.RESULTS_ROOT
        if self.local:
            self.RESULTS_ROOT = "/Users/jheitz/phd/results"
            # remote, but accessed from local machine
            self.RESULTS_ROOT_REMOTE = "/Volumes/methlab/Students/Jonathan/results"

        self.PREPROCESSED_DATA = "/home/ubuntu/methlab/Students/Jonathan/data_preprocessed"
        self.PREPROCESSED_DATA_REMOTE = self.PREPROCESSED_DATA
        if self.local:
            self.PREPROCESSED_DATA = "/Users/jheitz/phd/data_preprocessed"
            self.PREPROCESSED_DATA_REMOTE = "/Volumes/methlab/Students/Jonathan/data_preprocessed"

        # Segmentation files (timing for INV & PAR)
        self.DATA_ADReSS_SEGMENTATION = os.path.join(self.PREPROCESSED_DATA, "ADReSS_segmentation")

        # create an explicit attribute LOCAL & REMOTE to get the constants of the local or remote environment
        if recursion:
            # make sure this is not done recursively forever (only once -> recursion=False here)
            self.LOCAL = Constants(local=True, recursion=False)
            self.REMOTE = Constants(local=False, recursion=False)






