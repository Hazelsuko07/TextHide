export GLUE_DIR=./glue_data/

# RTE
export TASK_NAME=RTE
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 15.0 \
  --dropout 0 \
  --num_k 4 \
  --num_sigma 16 \
  --output_dir ./results/$TASK_NAME/BERT_16_4/ \
  --overwrite_output_dir

# MRPC
export TASK_NAME=MRPC
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --dropout 0 \
  --num_k 4 \
  --num_sigma 256 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir

# STS-B
export TASK_NAME=STS-B
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --dropout 0 \
  --num_k 4 \
  --num_sigma 256 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir

# CoLA
export TASK_NAME=CoLA
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 30.0 \
  --dropout 0 \
  --num_k 4 \
  --num_sigma 256 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir

# SST-2
export TASK_NAME=SST-2
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 20.0 \
  --num_k 4 \
  --dropout 0.4 \
  --num_sigma 256 \
  --save_steps 5000 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir

# QNLI
export TASK_NAME=QNLI
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 15.0 \
  --num_k 4 \
  --dropout 0.4 \
  --num_sigma 256 \
  --save_steps 5000 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir

# QQP
export TASK_NAME=QQP
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 15.0 \
  --num_k 4 \
  --dropout 0 \
  --num_sigma 256 \
  --save_steps 5000 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir

# MNLI
export TASK_NAME=MNLI
python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 15.0 \
  --num_k 4 \
  --num_sigma 256 \
  --save_steps 5000 \
  --output_dir ./results/$TASK_NAME/BERT_256_4/ \
  --overwrite_output_dir