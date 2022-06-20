#!/bin/bash

echo "-- Processing val set --"
python3 data/data_extractor.py \
  --bag_dir /home/plusai/code_space/plus.ai/data/snip_bag/ \
  --record_file /home/plusai/code_space/plus.ai/data/record_visual_test_bag.txt \
  --save_dir /home/plusai/code_space/plus.ai/data/features/ \
  --mode val \
  --argo \
  --obs_len 20 \
  --pred_len 30 \
