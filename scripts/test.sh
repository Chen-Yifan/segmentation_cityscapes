python prediction.py \
--dataset ./dataset/ \
--ckpt_path ./checkpoints/cityscapes_unet_SGD_randomcrop_60e/ \
--results_path ./results/cityscapes_unet_SGD_randomcrop_60e/ \
--epochs 54 \
--batch_size 1 \
--opt SGD \
--h 512 \
--w 1024 \
--split test \
