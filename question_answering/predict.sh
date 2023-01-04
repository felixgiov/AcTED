python run_classifier.py \
  --model_name_or_path ../tmp/roberta-large-mctaco-duration \
  --test_data ../tmp/event_qa_to_predict.tsv \
  --do_predict \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --output_dir ../tmp \
  --overwrite_output_dir