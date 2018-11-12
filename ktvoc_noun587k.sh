#!/bin/sh

USR_DIR=ktvoc
PROBLEM=ktvoc_noun587k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=/data/t2t_data/$PROBLEM
TMP_DIR=/data/t2t_datagen
TRAIN_DIR=/data/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
# * The following files should be stored in $TMP_DIR
#   * Data: kt.voc.noun.587k.pairs
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
WORKER_GPU=3  # The number of GPUs for training
export CUDA_VISIBLE_DEVICES=0,1,2  # GPU IDs to be used

t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --worker_gpu=$WORKER_GPU \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=2048'
