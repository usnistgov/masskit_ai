#!/bin/bash

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate ml_peptide

python train.py ms=tandem_lores experiment_name=test_prediction ms.max_mz=501 logging=mlflow_tensorboard_local ml.max_epochs=5 ml.limit_train_batches=500 +create_symlink=/tmp/best_test_model
python predict.py model_ensemble=[/tmp/best_test_model] num=100 output_prefix=test_prediction

