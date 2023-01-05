python run_classifier.py \
  --model_name_or_path roberta-large \
  --train_data ../data_generation/pseudo_qa_data/pseudo_qa_1000.tsv \
  --dev_data ../tmp/pseudo_qa_dev.tsv \
  --test_data ../tmp/pseudo_qa_dev.tsv \
  --do_train \
  --do_eval \
  --data_dir ../tmp \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --seed 321 \
  --output_dir ../tmp/roberta-large-pseudo-qa \
  --overwrite_output_dir