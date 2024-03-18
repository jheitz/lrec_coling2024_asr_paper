import stanza
import numpy as np
import os

from dataloader.dataset import TextDataset
from config.constants import Constants
from preprocessing.linguistic_features_literature import LinguisticFeaturesLiterature, LinguisticFeatureCalculator

class TestLinguisticFeaturesLiterature:
    constants = Constants()
    linguistic_features = LinguisticFeaturesLiterature(config={}, constants=constants)
    nlp_pipeline = stanza.Pipeline('en')

    def _get_feature_calculator(self, text):
        doc = self.nlp_pipeline(text.lower())
        feature_calculator = LinguisticFeatureCalculator(doc, constants=self.linguistic_features.CONSTANTS)
        return feature_calculator

    def _get_linguistic_features(self, text):
        return self.linguistic_features._calculate_for_text(text)
    def test_simple_features(self):
        text = "I don't know what I should say."
        feature_calculator = self._get_feature_calculator(text)
        assert feature_calculator.n_words() == 9  # including period and n't
        assert feature_calculator.n_unique() == 8

        text = "The boy is stealing the cookie. It is true."
        feature_calculator = self._get_feature_calculator(text)
        assert 3.18 < feature_calculator.word_length() < 3.19  # including period as word
        assert feature_calculator.sentence_length() == 5.5

    def test_pos_features(self):
        text = "I don't know what I should say."
        features = self._get_linguistic_features(text)
        assert features['pronoun_noun_ratio'] == 0  # no nouns -> 0
        assert features['verb_noun_ratio'] == 0  # no nouns -> 0

        text = "The boy is getting the cookie."
        features = self._get_linguistic_features(text)
        assert features['pronoun_noun_ratio'] == 0  # no pronouns -> 0
        assert features['verb_noun_ratio'] == 1/2

        text = "A hungry lion was hunting an innocent deer in the forest early in the morning to satisfy its hunger " \
               "and have fun if possible."
        # a(DET/DT) hungry(ADJ/JJ) lion(NOUN/NN) was(AUX/VBD) hunting(VERB/VBG) an(DET/DT) innocent(ADJ/JJ)
        # deer(NOUN/NN) in(ADP/IN) the(DET/DT) forest(NOUN/NN) early(ADV/RB) in(ADP/IN) the(DET/DT) morning(NOUN/NN)
        # to(PART/TO) satisfy(VERB/VB) its(PRON/PRP$) hunger(NOUN/NN) and(CCONJ/CC) have(VERB/VB) fun(NOUN/NN)
        # if(SCONJ/IN) possible(ADJ/JJ) .(PUNCT/.)
        features = self._get_linguistic_features(text)
        assert features['pronoun_noun_ratio'] == 1 / 6
        assert features['verb_noun_ratio'] == 3 / 6
        assert features['subordinate_coordinate_conjunction_ratio'] == 1
        assert features['adverb_ratio'] == 1 / 25
        assert features['noun_ratio'] == 6 / 25
        assert features['verb_ratio'] == 3 / 25
        assert features['pronoun_ratio'] == 1 / 25
        assert features['personal_pronoun_ratio'] == 0 / 25
        assert features['determiner_ratio'] == 4 / 25
        assert features['preposition_ratio'] == 3 / 25
        assert features['verb_present_participle_ratio'] == 1 / 25
        assert features['verb_modal_ratio'] == 0 / 25
        assert features['verb_third_person_singular_ratio'] == 0 / 25

    def test_words_not_in_dict(self):
        # Note that "is" is not considered since only words > 2 letters are relevant for this feature
        text = "The boy is blabling the cookie"
        feature_calculator = self._get_feature_calculator(text)
        assert feature_calculator.not_in_dictionary() == 0.2

        # some wave2vec2 transcription
        # should be all non-dictionary words but "ama" and "stor" are in the dictionary for some reason, okay...
        text = "AMA AAJAR  STOR STOARAS ATAA AIARTOR"
        feature_calculator = self._get_feature_calculator(text)
        assert feature_calculator.not_in_dictionary() == 4/6

