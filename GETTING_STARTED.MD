# Download weights
https://connectomics.clps.brown.edu/tf_records/zucker_weights/

# Download BSDS images
https://connectomics.clps.brown.edu/tf_records/bsds_test_images/

# Train a model on BSDS
CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=BSDS500_combos_100_extra_aux --no_db --model=gammanet_bsds

# Test on model on BSDS
CKPT=<path>/<to>/<weights>.ckpt
CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=BSDS500_test --model=gammanet_bsds --no_db --ckpt=$CKPT --placeholders --test --out_dir=$OUT_DIR --train=BSDS500_test_portrait --val=BSDS500_val_portrait
CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=BSDS500_test --model=gammanet_bsds --no_db --ckpt=$CKPT --placeholders --test --out_dir=$OUT_DIR --train=BSDS500_test_landscape --val=BSDS500_val_landscape

