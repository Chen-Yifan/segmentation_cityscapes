python train.py \
--dataset /media/exfat/yifan/dataset/ \
--ckpt_path /media/exfat/yifan/rf_checkpoints/cityscapes_unet_1024_Adam_100e/ \
--results_path /media/exfat/yifan/rf_results/cityscapes_unet_1024_Adam_100e/ \
--network Unet \
--epochs 100 \
--n_classes 34 \
--batch_size 2 \
--opt 1 \
--gpus 2
