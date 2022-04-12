# Train a model (e.g., \gamma-net) on a dataset (e.g., BSDS500)
0. Configure environment and download example dataset.
	- Copy `config.py.template` to `config.py` and fill in missing entries to reflect your system.
	- Do the same for `db/credentials.py`
	- run `python setup.py install`
	- Download the data files from http://bit.ly/bsds_tfrecords. Place the files in the `self.tf_records` directory that you specified in `config.py`.
	+ You can ignore errors with psql install and database creation.


1. CUDA_VISIBLE_DEVICES=7 python run_job.py --experiment=BSDS500_combos_100_extra_aux --no_db --model=gammanet_bsds --train=BSDS500_100_hed --val=BSDS500_100_jk 
2. bash in_silico_sims.sh  # Get neurophys recordings
3. bash in_silico_perfs.sh  # Get performances
4. bash finetune_models_seg_ae.sh


# Functional connectivity studies use the following repo <refactor_gammanet_connectivity>

