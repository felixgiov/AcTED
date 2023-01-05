python run_classifier.py \
  --model_name_or_path ../tmp/roberta-large-mctaco-duration \
  --train_data ../tmp/dev_3783_duration.tsv \
  --dev_data ../tmp/test_9442_duration.tsv \
  --test_data ../tmp/test_9442_duration.tsv \
  --do_train \
  --do_eval \
  --data_dir ../tmp \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --seed 321 \
  --output_dir ../tmp/roberta-large-pseudo-qa-then-mctaco-duration \
  --overwrite_output_dir