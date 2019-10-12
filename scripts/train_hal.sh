python train.py \
--dataset ./dataset/ \
--ckpt_path ./checkpoints/cityscapes_unet_1024_Adam_100e/ \
--results_path ./results/cityscapes_unet_1024_Adam_100e/ \
--network Unet \
--epochs 100 \
--n_classes 34 \
--batch_size 4 \
--opt 0 \
--gpus 3
