datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)

#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/gptv7_forward.yaml --runname gptv7_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/gptv7_literature_forward.yaml --runname gptv7_literature_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection

#python -u run.py --config ../configs/ADReSS/linguistic_features_liwc.yaml --runname linguistic_features_liwc --results_base_dir ${datetime_str}_liwc
#python -u run.py --config ../configs/ADReSS/linguistic_features_liwc+gpt.yaml --runname linguistic_features_liwc --results_base_dir ${datetime_str}_liwc


#python -u run.py --config ../configs/ADReSS/GPT3_finetuning_on_manual.yaml --runname GPT3_finetuning_on_manual
#exit

#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v6.yaml --runname linguistic_features_gpt_v6_google_speech
#python -u run.py --config ../configs/TAUKADIAL/liwc_gpt_train.yaml --runname taukadial_liwc_gpt
#exit



for tt in 1 2 3 4 5 6 7 8 9 10
do
  datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
  # run of all relevant settings for paper
  python -u run.py --config ../configs/ADReSS/gpt-paper/bert/BERT_on_google_speech_cv.yaml --runname BERT_on_google_speech_cv --results_base_dir ${datetime_str}_results_for_paper_bert
  python -u run.py --config ../configs/ADReSS/gpt-paper/bert/BERT_on_google_speech_test_on_testset.yaml --runname BERT_on_google_speech_test_on_testset --results_base_dir ${datetime_str}_results_for_paper_bert
  python -u run.py --config ../configs/ADReSS/gpt-paper/bert/BERT_on_manual_cv.yaml --runname BERT_on_manual_cv --results_base_dir ${datetime_str}_results_for_paper_bert
  python -u run.py --config ../configs/ADReSS/gpt-paper/bert/BERT_on_manual_test_on_testset.yaml --runname BERT_on_manual_test_on_testset --results_base_dir ${datetime_str}_results_for_paper_bert
  python -u run.py --config ../configs/ADReSS/gpt-paper/bert/BERT_on_whisper_cv.yaml --runname BERT_on_whisper_cv --results_base_dir ${datetime_str}_results_for_paper_bert
  python -u run.py --config ../configs/ADReSS/gpt-paper/bert/BERT_on_whisper_test_on_testset.yaml --runname BERT_on_whisper_test_on_testset --results_base_dir ${datetime_str}_results_for_paper_bert

  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-google-cv.yaml --runname gpt-finetuned-google-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-google-test_on_testset.yaml --runname gpt-finetuned-google-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-manual-cv.yaml --runname gpt-finetuned-manual-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-manual-test_on_testset.yaml --runname gpt-finetuned-manual-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-whisper-cv.yaml --runname gpt-finetuned-whisper-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-whisper-test_on_testset.yaml --runname gpt-finetuned-whisper-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned

  :
done

#for tt in 1 #2 3 4 5 6 7 8 9 10
#do
#  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-google-cv.yaml --runname gpt-finetuned-google-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-google-test_on_testset.yaml --runname gpt-finetuned-google-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-manual-cv.yaml --runname gpt-finetuned-manual-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-manual-test_on_testset.yaml --runname gpt-finetuned-manual-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-whisper-cv.yaml --runname gpt-finetuned-whisper-cv --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#  python -u run.py --config ../configs/ADReSS/gpt-paper/gpt-finetuned/gpt-finetuned-whisper-test_on_testset.yaml --runname gpt-finetuned-whisper-test_on_testset --results_base_dir ${datetime_str}_results_for_paper_gpt_finetuned
#  :
#done


#
#for tt in 1 #2 3 4 5 6 7 8 9 10
#do
##  #datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#  # run of all relevant settings for paper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v6_explainability.yaml --runname linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv6_explainability_literature.yaml --runname linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gptv6_explainability_literature.yaml --runname google_speech_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v6_explainability.yaml --runname google_speech_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v3.yaml --runname google_speech_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gptv3_literature.yaml --runname google_speech_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_literature.yaml --runname google_speech_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gptv6_explainability_literature.yaml --runname whisper_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gpt_v6_explainability.yaml --runname whisper_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gpt_v3.yaml --runname whisper_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gptv3_literature.yaml --runname whisper_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_literature.yaml --runname whisper_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_google_speech_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_google_speech__linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_google_speech__linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_google_speech__linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_google_speech__linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_whisper_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_whisper__linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_whisper__linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_whisper__linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_whisper__linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets
#
#  :
#done
#
#
### TEST ON ONLY TEST SET
#for tt in 1 #2 3 4 5 6 7 8 9 10
#do
##  #datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#  # run of all relevant settings for paper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/test_on_testset/linguistic_features_gpt_v6_explainability.yaml --runname linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/test_on_testset/linguistic_features_gptv6_explainability_literature.yaml --runname linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/test_on_testset/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/test_on_testset/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/test_on_testset/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/linguistic_features_gptv6_explainability_literature.yaml --runname google_speech_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/linguistic_features_gpt_v6_explainability.yaml --runname google_speech_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/linguistic_features_gpt_v3.yaml --runname google_speech_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/linguistic_features_gptv3_literature.yaml --runname google_speech_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/linguistic_features_literature.yaml --runname google_speech_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/linguistic_features_gptv6_explainability_literature.yaml --runname whisper_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/linguistic_features_gpt_v6_explainability.yaml --runname whisper_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/linguistic_features_gpt_v3.yaml --runname whisper_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/linguistic_features_gptv3_literature.yaml --runname whisper_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/linguistic_features_literature.yaml --runname whisper_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_google_speech_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_google_speech__linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_google_speech__linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_google_speech__linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/google_speech/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_google_speech__linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_whisper_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_whisper__linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_whisper__linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_whisper__linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/test_on_testset/whisper/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_whisper__linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_test_on_testset
#  :
#done
#
#
### CV ON ONLY TRAIN SET -> Model selection
#for tt in 1 #2 3 4 5 6 7 8 9 10
#do
##  #datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#  # run of all relevant settings for paper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/only_train/linguistic_features_gpt_v6_explainability.yaml --runname linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/only_train/linguistic_features_gptv6_explainability_literature.yaml --runname linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/only_train/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/only_train/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/only_train/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/linguistic_features_gptv6_explainability_literature.yaml --runname google_speech_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/linguistic_features_gpt_v6_explainability.yaml --runname google_speech_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/linguistic_features_gpt_v3.yaml --runname google_speech_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/linguistic_features_gptv3_literature.yaml --runname google_speech_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/linguistic_features_literature.yaml --runname google_speech_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/linguistic_features_gptv6_explainability_literature.yaml --runname whisper_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/linguistic_features_gpt_v6_explainability.yaml --runname whisper_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/linguistic_features_gpt_v3.yaml --runname whisper_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/linguistic_features_gptv3_literature.yaml --runname whisper_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/linguistic_features_literature.yaml --runname whisper_linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_google_speech_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_google_speech__linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_google_speech__linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_google_speech__linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/google_speech/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_google_speech__linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_whisper_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_whisper__linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_whisper__linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_whisper__linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/only_train/whisper/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_whisper__linguistic_features_literature --results_base_dir ${datetime_str}_feature_sets_only_train
#
#  :
#done




for tt in 1 2 3 4 5 6 7 8 9 10
do
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv6b_literature.yaml --runname linguistic_features_gptv6b_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v6b.yaml --runname linguistic_features_gpt_v6b --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv6_explainability0.5_literature.yaml --runname linguistic_features_gptv6_explainability0.5_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v6_explainability0.5.yaml --runname linguistic_features_gpt_v6_explainability0.5 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v6.yaml --runname linguistic_features_gpt_v6 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv6_literature.yaml --runname linguistic_features_gptv6_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv6_literature.yaml --runname linguistic_features_gptv6_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v7.yaml --runname linguistic_features_gpt_v7 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv7_literature.yaml --runname linguistic_features_gptv7_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v7b.yaml --runname linguistic_features_gpt_v7b --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv7b_literature.yaml --runname linguistic_features_gptv7b_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  #exit
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v4.yaml --runname linguistic_features_gpt_v4 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_differebnt_sets/linguistic_features_gptv4_literature.yaml --runname linguistic_features_gptv4_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v3_v4.yaml --runname linguistic_features_gpt_v3_v4 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v2.yaml --runname linguistic_features_gpt_v2 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v1.yaml --runname linguistic_features_gpt_v1 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv2_literature.yaml --runname linguistic_features_gptv2_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv1_literature.yaml --runname linguistic_features_gptv1_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv8_literature.yaml --runname linguistic_features_gptv8_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v8.yaml --runname linguistic_features_gpt_v8 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv9_literature.yaml --runname linguistic_features_gptv9_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v9.yaml --runname linguistic_features_gpt_v9 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv10_literature.yaml --runname linguistic_features_gptv10_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v10.yaml --runname linguistic_features_gpt_v10 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv10_subset_literature.yaml --runname linguistic_features_gptv10subset_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v10_subset.yaml --runname linguistic_features_gpt_v10subset --results_base_dir ${datetime_str}_linguistic_features_sets
  :
done



## all only on train, dont touch test set
#for tt in 1 #2 3 4 5 6 7 8 9 10
#do
#  #datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#  # run of all relevant settings for paper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv6b_literature.yaml --runname linguistic_features_gptv6b_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v6b.yaml --runname linguistic_features_gpt_v6b --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
#  #exit
##  #exit
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v6.yaml --runname linguistic_features_gpt_v6 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv6_literature.yaml --runname linguistic_features_gptv6_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v7.yaml --runname linguistic_features_gpt_v7 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv7_literature.yaml --runname linguistic_features_gptv7_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v7b.yaml --runname linguistic_features_gpt_v7b --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv7b_literature.yaml --runname linguistic_features_gptv7b_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  #exit
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v4.yaml --runname linguistic_features_gpt_v4 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv4_literature.yaml --runname linguistic_features_gptv4_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v3_v4.yaml --runname linguistic_features_gpt_v3_v4 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v2.yaml --runname linguistic_features_gpt_v2 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gpt_v1.yaml --runname linguistic_features_gpt_v1 --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv2_literature.yaml --runname linguistic_features_gptv2_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
##  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_trainval/linguistic_features_gptv1_literature.yaml --runname linguistic_features_gptv1_literature --results_base_dir ${datetime_str}_linguistic_features_sets_trainval
#done




for tt in 1 #2 3 4 5 6 7 8 9 10
do
  #datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#  # run of all relevant settings for paper
  #exit

#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gptv6_explainability_literature.yaml --runname linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v6_explainability.yaml --runname linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech

#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gptv6_literature.yaml --runname linguistic_features_gptv6_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v6.yaml --runname linguistic_features_gpt_v6 --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gptv6_explainability0.5_literature.yaml --runname linguistic_features_gptv6_explainability0.5_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/linguistic_features_gpt_v6_explainability0.5.yaml --runname linguistic_features_gpt_v6_explainability0.5 --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech


#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gptv6_explainability_literature.yaml --runname linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_linguistic_features_sets_whisper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gpt_v6_explainability.yaml --runname linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_linguistic_features_sets_whisper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_linguistic_features_sets_whisper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_linguistic_features_sets_whisper
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/whisper/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_linguistic_features_sets_whisper


#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gptv6_explainability_literature.yaml --runname ADR_PITT_linguistic_features_gptv6_explainability_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability.yaml --runname ADR_PITT_linguistic_features_gpt_v6_explainability --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v3.yaml --runname ADR_PITT_linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gptv3_literature.yaml --runname ADR_PITT_linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_literature.yaml --runname ADR_PITT_linguistic_features_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech

#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gptv6_literature.yaml --runname ADR_PITT_linguistic_features_gptv6_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v6.yaml --runname ADR_PITT_linguistic_features_gpt_v6 --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gptv6_explainability0.5_literature.yaml --runname ADR_PITT_linguistic_features_gptv6_explainability0.5_literature --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets_asr/google_speech/ADReSS_with_PITT/linguistic_features_gpt_v6_explainability0.5.yaml --runname ADR_PITT_linguistic_features_gpt_v6_explainability0.5 --results_base_dir ${datetime_str}_linguistic_features_sets_google_speech

#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v6.yaml --runname linguistic_features_gpt_v6 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv6_literature.yaml --runname linguistic_features_gptv6_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v7.yaml --runname linguistic_features_gpt_v7 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv7_literature.yaml --runname linguistic_features_gptv7_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v7b.yaml --runname linguistic_features_gpt_v7b --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv7b_literature.yaml --runname linguistic_features_gptv7b_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  #exit
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v4.yaml --runname linguistic_features_gpt_v4 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv4_literature.yaml --runname linguistic_features_gptv4_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v3_v4.yaml --runname linguistic_features_gpt_v3_v4 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v2.yaml --runname linguistic_features_gpt_v2 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v1.yaml --runname linguistic_features_gpt_v1 --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv2_literature.yaml --runname linguistic_features_gptv2_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#  python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv1_literature.yaml --runname linguistic_features_gptv1_literature --results_base_dir ${datetime_str}_linguistic_features_sets
  :
done




# some hyperparameter testing for RandomForest, as there is significant overfitting
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100.yaml --runname RF_100 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_3.yaml --runname RF_100_3 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_5.yaml --runname RF_100_5 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_10.yaml --runname RF_100_10 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_15.yaml --runname RF_100_15 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_20.yaml --runname RF_100_20 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_30.yaml --runname RF_100_30 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_100_50.yaml --runname RF_100_50 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500.yaml --runname RF_500 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_3.yaml --runname RF_500_3 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_5.yaml --runname RF_500_5 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_10.yaml --runname RF_500_10 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_15.yaml --runname RF_500_15 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_20.yaml --runname RF_500_20 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_30.yaml --runname RF_500_30 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_50.yaml --runname RF_500_50 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_1000.yaml --runname RF_1000 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_1000_3.yaml --runname RF_1000_3 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_1000_5.yaml --runname RF_1000_5 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_1000_10.yaml --runname RF_1000_10 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_2000.yaml --runname RF_2000 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_2000_3.yaml --runname RF_2000_3 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_2000_5.yaml --runname RF_2000_5 --results_base_dir ${datetime_str}_random_forest_hyperparameters
#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_2000_10.yaml --runname RF_2000_10 --results_base_dir ${datetime_str}_random_forest_hyperparameters

#python -u run.py --config ../configs/ADReSS/random_forest_hyperparameters/RF_500_5_5.yaml --runname RF_500_5_5 --results_base_dir ${datetime_str}_random_forest_hyperparameters





#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_best_gpt.yaml --runname linguistic_features_best_gpt --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_features.yaml --runname linguistic_features_gpt_features --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_best_without_gpt.yaml --runname linguistic_features_best_without_gpt --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_all.yaml --runname linguistic_features_all --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_best.yaml --runname linguistic_features_best_with_gpt --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v1.yaml --runname linguistic_features_gpt_v1 --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v2.yaml --runname linguistic_features_gpt_v2 --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v4b.yaml --runname linguistic_features_gpt_v4b --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_all_xgboost.yaml --runname linguistic_features_all_xgboost --results_base_dir ${datetime_str}_linguistic_features_sets

#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v3.yaml --runname linguistic_features_gpt_v3 --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v4.yaml --runname linguistic_features_gpt_v4 --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv4_literature.yaml --runname linguistic_features_gptv4_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gptv3_literature.yaml --runname linguistic_features_gptv3_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_linguistic_features_sets
#python -u run.py --config ../configs/ADReSS/linguistic_features_different_sets/linguistic_features_gpt_v3_v4.yaml --runname linguistic_features_gpt_v3_v4 --results_base_dir ${datetime_str}_linguistic_features_sets


#python -u run.py --config ../configs/ADReSS/linguistic_features.yaml --runname linguistic_features
#python -u run.py --config ../configs/ADReSS/linguistic_features_gpt_features.yaml --runname linguistic_features_gpt_features_v1_v2_temp05

#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/all_backward_rfe.yaml --runname all_backward_rfe --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/all_backward.yaml --runname all_backward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/all_forward.yaml --runname all_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/literature_features_backward.yaml --runname literature_features_backward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/literature_features_forward.yaml --runname literature_features_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/all_except_gpt_forward.yaml --runname all_except_gpt_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/gpt_forward.yaml --runname gpt_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/gpt_backward.yaml --runname gpt_backward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/gptv4_forward.yaml --runname gptv4_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/gptv4_literature_forward.yaml --runname gptv4_literature_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection
#python -u run.py --config ../configs/ADReSS/linguistic_features_feature_selection/literature_features_forward.yaml --runname literature_features_forward --results_base_dir ${datetime_str}_linguistic_features_feature_selection


#python -u run.py --config ../configs/ADReSS/gpt/GPT3_finetuning.yaml --runname GPT3_finetuning
#python -u run.py --config ../configs/ADReSS/gpt/GPT3_finetuning_newsplit.yaml --runname GPT3_finetuning_newsplit
#python -u run.py --config ../configs/ADReSS/gpt/GPT3_finetuning_newsplit_original.yaml --runname GPT3_finetuning_newsplit_original


#for tt in 1 2 3 4 5 6 7 8
#do
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_5_4e-4.yaml --runname BERT_on_manual_5_4e-4 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_5_4e-5.yaml --runname BERT_on_manual_5_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_5_4e-6.yaml --runname BERT_on_manual_5_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_8_2e-5.yaml --runname BERT_on_manual_8_2e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_10_4e-4.yaml --runname BERT_on_manual_10_4e-4 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_10_4e-5.yaml --runname BERT_on_manual_10_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_10_4e-6.yaml --runname BERT_on_manual_10_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_20_4e-5.yaml --runname BERT_on_manual_20_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_20_4e-6.yaml --runname BERT_on_manual_20_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_20_4e-7.yaml --runname BERT_on_manual_20_4e-7 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_30_1e-6.yaml --runname BERT_on_manual_30_1e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_30_1e-7.yaml --runname BERT_on_manual_30_1e-7 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_30_4e-5.yaml --runname BERT_on_manual_30_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_30_4e-6.yaml --runname BERT_on_manual_30_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/BERT_on_manual_30_4e-7.yaml --runname BERT_on_manual_30_4e-7 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_5_4e-4.yaml --runname BERT_large_on_manual_5_4e-4 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_5_4e-5.yaml --runname BERT_large_on_manual_5_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_5_4e-6.yaml --runname BERT_large_on_manual_5_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_8_2e-5.yaml --runname BERT_large_on_manual_8_2e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_10_4e-4.yaml --runname BERT_large_on_manual_10_4e-4 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_10_4e-5.yaml --runname BERT_large_on_manual_10_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_10_4e-6.yaml --runname BERT_large_on_manual_10_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_20_4e-5.yaml --runname BERT_large_on_manual_20_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_20_4e-6.yaml --runname BERT_large_on_manual_20_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_20_4e-7.yaml --runname BERT_large_on_manual_20_4e-7 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_30_1e-6.yaml --runname BERT_large_on_manual_30_1e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_30_1e-7.yaml --runname BERT_large_on_manual_30_1e-7 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_30_4e-5.yaml --runname BERT_large_on_manual_30_4e-5 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_30_4e-6.yaml --runname BERT_large_on_manual_30_4e-6 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_yuanetal/bert-large/BERT_on_manual_30_4e-7.yaml --runname BERT_large_on_manual_30_4e-7 --results_base_dir ${datetime_str}_overview_yuanetal_pause_coding
#done


#python -u run.py --config ../configs/ADReSS/BERT_on_manual_pause_coding.yaml --runname BERT_on_manual_pause_coding

#python -u run.py --config ../configs/ADReSS/train-validation-test-split/GPT_prompt_v7.yaml --runname gpt_prompt_v7
#python -u run.py --config ../configs/ADReSS/train-validation-test-split/GPT_prompt_v9_no_background.yaml --runname gpt_prompt_v9

#python -u run.py --config ../configs/ADReSS/train-validation-test-split/BERT_on_manual.yaml --runname BERT_on_manual-train-val-test #--results_base_dir ${datetime_str}_whisper-swiss-german
#python -u run.py --config ../configs/ADReSS/train-validation-test-split/linguistic_features_literature.yaml --runname linguistic_features_literature-train-val-test #--results_base_dir ${datetime_str}_whisper-swiss-german

#python -u run.py --config ../configs/ADReSS/train-test-split/BERT_on_manual.yaml --runname BERT_on_manual-train-test #--results_base_dir ${datetime_str}_whisper-swiss-german
#python -u run.py --config ../configs/ADReSS/train-test-split/linguistic_features_literature.yaml --runname linguistic_features_literature-train-test #--results_base_dir ${datetime_str}_whisper-swiss-german

#python -u run.py --config ../configs/VELAS/whisper-swiss-german.yaml --runname whisper-swiss-german #--results_base_dir ${datetime_str}_whisper-swiss-german

#for t in 1 2 3 4 5 6 7 8:
#do
#  python -u run.py --config ../configs/ADReSS/tests/BERT_on_manual_old_chat_parser_version.yaml --runname BERT_on_manual_old_chat_parser_version --results_base_dir ${datetime_str}_chat_parser
#  python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --runname BERT_on_manual --results_base_dir ${datetime_str}_chat_parser
#  :e
#done
#python -u run.py --config ../configs/ADReSS/linguistic_features_literature_INV_PAR.yaml --runname linguistic_features_literature



#for tt in 1 2 3 4 5 6 7 8 9 10
#do
  #datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
  # run of all relevant settings for paper
  #python -u run.py --config ../configs/ASR_paper/linguistic_features_literature.yaml --runname linguistic_features_literature --results_base_dir ${datetime_str}_overview_with_interviewer
  #python -u run.py --config ../configs/ASR_paper/ling_feat_lit_google_speech_PAR_only.yaml --runname ling_feat_lit_google_speech_PAR_only --results_base_dir ${datetime_str}_overview_with_interviewer
  #python -u run.py --config ../configs/ASR_paper/ling_feat_lit_wave2vec2_PAR_only.yaml --runname ling_feat_lit_wave2vec2_PAR_only --results_base_dir ${datetime_str}_overview_with_interviewer
  #python -u run.py --config ../configs/ASR_paper/ling_feat_lit_whisper_PAR_only.yaml --runname ling_feat_lit_whisper_PAR_only --results_base_dir ${datetime_str}_overview_with_interviewer
  #for t in 1 2 3 4 5 6 7 8:
  #do
    #python -u run.py --config ../configs/ASR_paper/BERT_on_google_speech_PAR_only.yaml --results_base_dir ${datetime_str}_overview_with_interviewer --runname BERT_on_google_speech_PAR_only
    #python -u run.py --config ../configs/ASR_paper/BERT_on_manual.yaml --runname BERT_on_manual --results_base_dir ${datetime_str}_overview_with_interviewer
    #python -u run.py --config ../configs/ASR_paper/BERT_on_wave2vec2_PAR_only.yaml --results_base_dir ${datetime_str}_overview_with_interviewer --runname BERT_on_wave2vec2_PAR_only
    #python -u run.py --config ../configs/ASR_paper/BERT_on_whisper_PAR_only.yaml --results_base_dir ${datetime_str}_overview_with_interviewer --runname BERT_on_whisper_PAR_only
  #done
#
#  python -u run.py --config ../configs/ADReSS/ling_feat_lit_google_speech.yaml --runname ling_feat_lit_google_speech --results_base_dir ${datetime_str}_overview_with_interviewer
#  python -u run.py --config ../configs/ADReSS/ling_feat_lit_wave2vec2.yaml --runname ling_feat_lit_wave2vec2 --results_base_dir ${datetime_str}_overview_with_interviewer
#  python -u run.py --config ../configs/ADReSS/ling_feat_lit_whisper.yaml --runname ling_feat_lit_whisper --results_base_dir ${datetime_str}_overview_with_interviewer
#  python -u run.py --config ../configs/ADReSS/linguistic_features_literature_INV_PAR.yaml --runname linguistic_features_literature_INV_PAR --results_base_dir ${datetime_str}_overview_with_interviewer
#  for t in 1 2 3 4 5 6 7 8:
#  do
#    python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_manual_INV_PAR.yaml --results_base_dir ${datetime_str}_overview_with_interviewer --runname BERT_on_manual_INV_PAR
#    python -u run.py --config ../configs/ADReSS/BERT_on_wave2vec2.yaml --runname BERT_on_wave2vec2 --results_base_dir ${datetime_str}_overview_with_interviewer
#    python -u run.py --config ../configs/ADReSS/BERT_on_whisper.yaml --runname BERT_on_whisper --results_base_dir ${datetime_str}_overview_with_interviewer
#    python -u run.py --config ../configs/ADReSS/BERT_on_google_speech.yaml --runname BERT_on_google_speech --results_base_dir ${datetime_str}_overview_with_interviewer
#  done
#done


## test effect of Interviewer sequence removal
#datetime_str=$(date '+%Y%m%d_%H%M')
#for t in 1 2 3 4 5 6 7 8
#do
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_google_speech_PAR_only.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_google_speech_PAR_only
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_google_speech.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_google_speech
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_manual.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_manual
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_manual_INV_PAR.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_manual_INV_PAR
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_wave2vec2.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_wave2vec2
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_wave2vec2_PAR_only.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_wave2vec2_PAR_only
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_whisper.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_whisper
#  python -u run.py --config ../configs/ADReSS/interviewer_influence/BERT_on_whisper_PAR_only.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_whisper_PAR_only
#done

#python -u run.py --config ../configs/test.yaml --runname test

#python -u run.py --config ../configs/asr/ADReSS_google_speech_v2_chirp.yaml --runname ADReSS_google_speech_v2_chirp --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_google_speech_v2_long.yaml --runname ADReSS_google_speech_v2_long --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_google_speech_v1.yaml --runname ADReSS_google_speech_v1 --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_wave2vec2.yaml --runname ADReSS_wave2vec2 --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_whisper.yaml --runname ADReSS_whisper --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_with_PITT_wave2vec2.yaml --runname ADReSS_with_PITT_wave2vec2 --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_with_PITT_whisper.yaml --runname ADReSS_with_PITT_whisper --results_base_dir ${datetime_str}_asr
#python -u run.py --config ../configs/asr/ADReSS_with_PITT_google_speech_v2_chirp.yaml --runname ADReSS_with_PITT_google_speech_v2_chirp --results_base_dir ${datetime_str}_asr

#python -u run.py --config ../configs/ADReSS/linguistic_features_comparison/linguistic_features_all.yaml --runname linguistic_features_all --results_base_dir ${datetime_str}_lingu_features_comparison
#python -u run.py --config ../configs/ADReSS/linguistic_features_comparison/linguistic_features__stanza_features.yaml --runname linguistic_features__stanza_features --results_base_dir ${datetime_str}_lingu_features_comparison
#python -u run.py --config ../configs/ADReSS/linguistic_features_comparison/linguistic_features_disfluency_features.yaml --runname linguistic_features_disfluency_features --results_base_dir ${datetime_str}_lingu_features_comparison
#python -u run.py --config ../configs/ADReSS/linguistic_features_comparison/linguistic_features_liwc_features_all.yaml --runname linguistic_features_liwc_features_all --results_base_dir ${datetime_str}_lingu_features_comparison
#python -u run.py --config ../configs/ADReSS/linguistic_features_comparison/linguistic_features_R_features.yaml --runname linguistic_features_R_features --results_base_dir ${datetime_str}_lingu_features_comparison
#python -u run.py --config ../configs/ADReSS/linguistic_features_comparison/linguistic_features_all_no_LIWC.yaml --runname linguistic_features_all_no_LIWC --results_base_dir ${datetime_str}_lingu_features_comparison


#python -u run.py --config ../configs/ADReSS/linguistic_audio_features.yaml --runname linguistic_audio_features --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/linguistic_audio_features+BERT.yaml --runname linguistic_audio_features+BERT --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/linguistic_features+BERT.yaml --runname linguistic_features+BERT --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/linguistic_features.yaml --runname linguistic_features --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/audio_features_LogReg.yaml --runname audio_features_LogReg --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS_with_PITT/audio_features_LogReg.yaml --runname audio_features_LogReg_ADR_PITT --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/linguistic_audio_features_ADR_ADR-PITT.yaml --runname linguistic_audio_features_ADR_ADR-PITT --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/linguistic_audio_features_ADR_ADR-PITT+BERT.yaml --runname linguistic_audio_features_ADR_ADR-PITT+BERT.yaml --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/linguistic_features+BERT_logreg.yaml --runname linguistic_features+BERT_logreg --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/linguistic_features_logreg.yaml --runname linguistic_features_logreg --results_base_dir ${datetime_str}_lingu_audio_features
#python -u run.py --config ../configs/ADReSS/audio_features_RF.yaml --runname audio_features_RF --results_base_dir ${datetime_str}_lingu_audio_features

## run linguistic features multiple times to see if it's consistent
#for t in 1 2 3 4 5 6 7 8:
#do
#  python -u run.py --config ../configs/ADReSS/linguistic_features.yaml --runname linguistic_features_${t} --results_base_dir ${datetime_str}_linguistic_features
#done

# run of all relevant settings, for good comparison of current performance
#for t in 1 2 3 4 5 6 7 8:
#do
#  python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --runname BERT_on_manual_ADR_PITT --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_wave2vec2.yaml --runname BERT_on_wave2vec2_ADR_PITT --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_whisper.yaml --runname BERT_on_whisper_ADR_PITT --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_google_speech.yaml --runname BERT_on_google_speech_ADR_PITT --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --runname BERT_on_manual --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS/BERT_on_wave2vec2.yaml --runname BERT_on_wave2vec2 --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS/BERT_on_whisper.yaml --runname BERT_on_whisper --results_base_dir ${datetime_str}_overview
#  python -u run.py --config ../configs/ADReSS/BERT_on_google_speech.yaml --runname BERT_on_google_speech --results_base_dir ${datetime_str}_overview
#done
#python -u run.py --config ../configs/ADReSS/linguistic_features.yaml --runname linguistic_features --results_base_dir ${datetime_str}_overview
#python -u run.py --config ../configs/ADReSS/linguistic_features+BERT.yaml --runname linguistic_features+BERT --results_base_dir ${datetime_str}_overview
#python -u run.py --config ../configs/ADReSS/punct+linguistic_features.yaml --runname punct+linguistic_features --results_base_dir ${datetime_str}_overview
#python -u run.py --config ../configs/ADReSS/punct+linguistic_features+BERT.yaml --runname punct+linguistic_features+BERT --results_base_dir ${datetime_str}_overview
#python -u run.py --config ../configs/ADReSS/linguistic_audio_features.yaml --runname linguistic_audio_features --results_base_dir ${datetime_str}_overview
#python -u run.py --config ../configs/ADReSS/linguistic_audio_features+BERT.yaml --runname linguistic_audio_features+BERT --results_base_dir ${datetime_str}_overview

# BERT on manual default settings -> to be used with linguistic features in RF as a feature
#for t in 1 2 3 4 5 6 7 8:
#do
#  python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --runname BERT_on_manual --results_base_dir ${datetime_str}_BERT_on_manual
#done


#python -u run.py --config ../configs/ADReSS/linguistic_features.yaml --runname linguistic_features --results_base_dir ${datetime_str}_linguistic_features
#python -u run.py --config ../configs/ADReSS/linguistic_features+BERT.yaml --runname linguistic_features+BERT --results_base_dir ${datetime_str}_linguistic_features
#python -u run.py --config ../configs/ADReSS/punct+linguistic_features.yaml --runname punct+linguistic_features --results_base_dir ${datetime_str}_automatic_punctuation
#python -u run.py --config ../configs/ADReSS/punct+linguistic_features+BERT.yaml --runname punct+linguistic_features+BERT --results_base_dir ${datetime_str}_automatic_punctuation

# some hyperparameter testing for dataloader of manual transcripts
#for t in 1 2 3 4 5 6 7 8:
#do
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par1_pause0_term1_unint0.yaml --runname BERT_on_manual_par1_pause0_term1_unint0 --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par1_pause1_term1_unint0.yaml --runname BERT_on_manual_par1_pause1_term1_unint0 --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par0_pause0_term1_unint0.yaml --runname BERT_on_manual_par0_pause0_term1_unint0 --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par1_pause1_term0_unint0_punctuation.yaml --runname BERT_on_manual_par1_pause1_term0_unint0_punctuation --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par1_pause0_term0_unint0_punctuation.yaml --runname BERT_on_manual_par1_pause0_term0_unint0_punctuation --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par1_pause0_term0_unint1.yaml --runname BERT_on_manual_par1_pause0_term0_unint1 --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
  #python -u run.py --config ../configs/ADReSS/hyperparameters_manual_transcripts/BERT_on_manual_par1_pause0_term1_unint1.yaml --runname BERT_on_manual_par1_pause0_term1_unint1 --results_base_dir ${datetime_str}_BERT_on_manual_hyperparameters
#done



#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_100_0.1.yaml --runname GB_100_0.1 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_100_0.01.yaml --runname GB_100_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_500_0.1.yaml --runname GB_500_0.1 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_500_0.01.yaml --runname GB_500_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_1000_0.1.yaml --runname GB_1000_0.1 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_1000_0.01.yaml --runname GB_1000_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_3000_0.01.yaml --runname GB_3000_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/GB_3000_0.001.yaml --runname GB_3000_0.001 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/RF_100.yaml --runname RF_100 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/RF_500.yaml --runname RF_500 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/RF_1000.yaml --runname RF_1000 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic/RF_3000.yaml --runname RF_3000 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_100_0.1.yaml --runname GB_100_0.1 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_100_0.01.yaml --runname GB_100_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_500_0.1.yaml --runname GB_500_0.1 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_500_0.01.yaml --runname GB_500_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_1000_0.1.yaml --runname GB_1000_0.1 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_1000_0.01.yaml --runname GB_1000_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_3000_0.01.yaml --runname GB_3000_0.01 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/GB_3000_0.001.yaml --runname GB_3000_0.001 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/RF_100.yaml --runname RF_100 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/RF_500.yaml --runname RF_500 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/RF_1000.yaml --runname RF_1000 --results_base_dir ${datetime_str}_hyperparameter_tree_based
#python -u run.py --config ../configs/ADReSS/hyperparameters_tree_based/linguistic_BERT/RF_3000.yaml --runname RF_3000 --results_base_dir ${datetime_str}_hyperparameter_tree_based

# multipe runs for ensemble LIME analysis
#for t in 1 2 3 4 5 6 7 8:
#do
#python -u run.py --config ../configs/ADReSS/BERT_on_manual_store_model_no_cv.yaml --runname BERT_on_manual_store_model_no_cv --results_base_dir ${datetime_str}_BERT_on_manual_store_model_no_cv
#done

#python -u run.py --config ../configs/ADReSS/linguistic_features.yaml --runname linguistic_features --results_base_dir ${datetime_str}_linguistic_vs_BERT
#python -u run.py --config ../configs/ADReSS/linguistic_features+BERT.yaml --runname linguistic_features_BERT #--results_base_dir ${datetime_str}_linguistic_vs_BERT
#for t in 1 2 3 4 5 6 7 8
#do
#  python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_linguistic_vs_BERT --runname BERT_on_manual
#done

# store model, without CV -> so training on entire train set -> this should give better LIME explainability
# python -u run.py --config ../configs/ADReSS/BERT_on_manual_store_model_no_cv.yaml --runname BERT_on_manual_store_model_no_cv


# test effect of Interviewer sequence removal
datetime_str=$(date '+%Y%m%d_%H%M')
#for t in 1 2 3 4 5 6 7 8
#do
  # new experiments july 24
#  python -u run.py --config ../configs/ADReSSo/BERT_on_wave2vec2_store_model.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_wave2vec2_store_model
#  python -u run.py --config ../configs/ADReSSo/BERT_on_whisper_store_model.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_whisper_store_model
#  python -u run.py --config ../configs/ADReSS/BERT_on_manual_INV_PAR_store_model.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_manual_INV_PAR_store_model

#	python -u run.py --config ../configs/ADReSSo/BERT_on_wave2vec2.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_wave2vec2
# python -u run.py --config ../configs/ADReSSo/BERT_on_wave2vec2_PAR_only.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_wave2vec2_PAR_only
# python -u run.py --config ../configs/ADReSSo/BERT_on_whisper.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_whisper
#python -u run.py --config ../configs/ADReSSo/BERT_on_whisper_PAR_only.yaml --results_base_dir ${datetime_str}_interviewer_influence --runname BERT_on_whisper_PAR_only
#	:
#done


# python -u run.py --config ../configs/ADReSS/BERT_on_wave2vec2.yaml --results_base_dir ${datetime_str}_asr_vs_manual_sample_names --runname BERT_on_wave2vec2_ADReSS
# python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_whisper.yaml --results_base_dir ${datetime_str}_asr_vs_manual_sample_names --runname BERT_on_whisper_PITT_raw
# python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_wave2vec2.yaml --results_base_dir ${datetime_str}_asr_vs_manual_sample_names --runname BERT_on_wave2vec2_PITT_raw

# python -u run.py --config ../configs/test.yaml --runname test_whisper

# python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --runname BERT_on_manual_store_sample_names
# python -u run.py --config ../configs/ADReSS/BERT_on_manual_store_model.yaml --runname BERT_on_manual_store_model

## test different learning rate schemes
#for t in 1 2 3 4 5 6 7 8 9
#do
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing_lr_sched/BERT_on_manual_linear.yaml --results_base_dir ${datetime_str}_learning_rate_scheme --runname BERT_on_manual_linear
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing_lr_sched/BERT_on_manual_LLRD.yaml --results_base_dir ${datetime_str}_learning_rate_scheme --runname BERT_on_manual_LLRD
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing_lr_sched/BERT_on_manual_LLRD_with_warmup.yaml --results_base_dir ${datetime_str}_learning_rate_scheme --runname BERT_on_manual_LLRD_with_warmup
#	:
#done


# test different transcription services
#datetime_str=$(date '+%Y%m%d_%H%M')
#for t in 1 2 3 4 5 6 7 8
#do
#	#python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_wave2vec2.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_wave2vec2_PITT_raw
#	python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_whisper.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_whisper_PITT_raw
#	python -u run.py --config ../configs/ADReSS/BERT_on_wave2vec2.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_wave2vec2_ADReSS
#	#python -u run.py --config ../configs/ADReSS/BERT_on_whisper.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_whisper_ADReSS
#	python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_manual
#	#python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_wave2vec2_ngrams.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_wave2vec2_ngrams_PITT_raw
#	#python -u run.py --config ../configs/ADReSS/BERT_on_wave2vec2_ngrams.yaml --results_base_dir ${datetime_str}_asr_vs_manual --runname BERT_on_wave2vec2_ngrams_ADReSS
#	:
#done


#python -u run.py --config ../configs/ADReSS/GPT3_on_manual.yaml --runname GPT3_on_manual
#
#for t in 1 2 3
#do
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_16_0.2.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_16_0.2
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_16_0.05.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_16_0.05
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_25_0.1.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_25_0.1
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_25_0.2.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_25_0.2
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_25_0.02.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_25_0.02
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_25_0.05.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_25_0.05
#
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_4_0.1.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_4_0.1
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_4_0.2.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_4_0.2
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_4_0.02.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_4_0.02
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_4_0.05.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_4_0.05
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_8_0.1.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_8_0.1
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_8_0.2.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_8_0.2
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_8_0.02.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_8_0.02
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_8_0.05.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_8_0.05
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_16_0.1.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_16_0.1
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing_gpt3/GPT3_on_manual_16_0.02.yaml --results_base_dir ${datetime_str}_gp3_hyperparameter_testing --runname GPT3_on_manual_16_0.02
#  :
#done



#for t in 1 2 3 4 5 6
#do
#  python -u run.py --config ../configs/ADReSS/BERT_on_wave2vec2.yaml --runname BERT_on_wave2vec2 --results_base_dir ${datetime_str}_BERT_on_wave2vec2
#  :
#done

# hyperparameter tests for epochs and learning rate of fine-tuning
#for t in 1 2 3 4 5 6
#do
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_30_4e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_30_4e-6
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_30_4e-7.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_30_4e-7
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_5_4e-4.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_5_4e-4
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_5_4e-5.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_5_4e-5
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_5_4e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_5_4e-6
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_10_4e-4.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_10_4e-4
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_10_4e-5.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_10_4e-5
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_10_4e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_10_4e-6
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_20_4e-5.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_20_4e-5
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_20_4e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_20_4e-6
#	python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_20_4e-7.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_20_4e-7
#	:
#done

#for t in 1 2 3 4 5 6
#do
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_40_4e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_40_4e-6
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_40_4e-7.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_40_4e-7
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_50_4e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_50_4e-6
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_50_4e-7.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_50_4e-7
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_30_1e-6.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_30_1e-6
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_30_1e-7.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_30_1e-7
#  python -u run.py --config ../configs/ADReSS/hyperparameter_testing/BERT_on_manual_30_4e-5.yaml --results_base_dir ${datetime_str}_hyperparameter_testing --runname BERT_on_manual_30_4e-5
#  :
#done

## extensive testing of same setup
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#python -u run.py --config ../configs/ADReSS/BERT_on_manual.yaml --results_base_dir ${datetime_str}_stability_tests_BERT_on_manual --runname BERT_on_manual
#
# python -u run.py --config ../configs/BERT_on_wave2vec2.yaml --runname BERT_on_wave2vec2
# python -u run.py --config ../configs/BERT_on_whisper.yaml --runname BERT_on_whisper

# python -u run.py --config ../configs/ADReSSo/BERT_on_wave2vec2.yaml --runname ADReSSo_BERT_on_wave2vec2
# python -u run.py --config ../configs/ADReSSo/BERT_on_whisper.yaml --runname ADReSSo_BERT_on_whisper
# python -u run.py --config ../configs/ADReSSo/BERT_on_whisper_PAR_only.yaml --runname ADReSSo_BERT_on_whisper_PAR_only
# python -u run.py --config ../configs/ADReSSo/BERT_on_wave2vec2_PAR_only.yaml --runname ADReSSo_BERT_on_wave2vec2_PAR_only

# python -u run.py --config ../configs/ADReSS_with_PITT/BERT_on_whisper.yaml --runname ADReSS_with_PITT_BERT_on_whisper

# python -u run.py --config ../configs/ADReSS/preprocessing_wave2vec2.yaml --runname ADReSS_wave2vec2 --results_base_dir ${datetime_str}_wave2vec2_transcriptions
# python -u run.py --config ../configs/ADReSS_with_PITT/preprocessing_wave2vec2.yaml --runname ADReSS_original_PITT_wave2vec2 --results_base_dir ${datetime_str}_wave2vec2_transcriptions
# python -u run.py --config ../configs/ADReSS/preprocessing_wave2vec2_ngrams.yaml --runname ADReSS_wave2vec2_ngrams
# python -u run.py --config ../configs/ADReSS_with_PITT/preprocessing_wave2vec2_ngrams.yaml --runname ADReSS_with_PITT_wave2vec2_ngrams
# python -u run.py --config ../configs/ADReSS/preprocessing_wave2vec2_ngrams_indomain.yaml --runname ADReSS_wave2vec2_ngrams_indomain --results_base_dir ${datetime_str}_wave2vec2_ngrams_indomain
# python -u run.py --config ../configs/ADReSS_with_PITT/preprocessing_wave2vec2_ngrams_indomain.yaml --runname ADReSS_with_PITT_wave2vec2_ngrams_indomain --results_base_dir ${datetime_str}_wave2vec2_ngrams_indomain

#python -u run.py --config ../configs/test.yaml --runname test_run
