#!/bin/bash
torchrun --nproc_per_node=1 multiple_train.py --save_path log-0 \
                                              --train_path /home/plusai/code_space/dataset/features/train/ \
                                              --val_path /home/plusai/code_space/dataset/features/sampled_val/ \
                                              --epoch 50 \
                                              --batch_size 2 \
                                              --workers 0 \
                                              --devices 0 \
                                              --env ATDSNetv0.1 \
                                              --record \
                                              --mode base
echo "finished one training"