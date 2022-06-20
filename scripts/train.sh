#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=1 multiple_train.py --save_path log-0 \
                                                                        --epoch 50 \
                                                                        --batch_size 4 \
                                                                        --workers 0 \
                                                                        --devices 0 \
                                                                        --env ATDSNetv0.1
echo "finished one training"