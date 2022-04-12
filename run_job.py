import os
import argparse
import numpy as np
from utils import logger
from utils import py_utils
from config import Config
from ops import model_tools
from plot_bsds_function import main as plotter
try:
    from db import db
except Exception as e:
    print('Failed to import db in run_job.py: %s' % e)


def get_fine_tune_params(**kwargs):
    """Parameters for fine-tuning, to e.g. test transfer/forgetting."""
    if kwargs is not None:
        default = {}
        for k, v in kwargs.iteritems():
            default[k] = v
        return default


def main(
        experiment,
        model,
        train,
        val,
        checkpoint,
        use_db=True,
        test=False,
        reduction=0,
        random=True,
        add_config=None,
        gpu_device=['/gpu:0'],
        cpu_device='/cpu:0',
        num_gpus=False,
        transfer=False,
        placeholders=False,
        save_test_npz=True,
        num_batches=None,
        map_out='test_maps',
        viz_ae=False,
        out_dir=None):
    """Interpret and run a model."""
    main_config = Config()
    dt_string = py_utils.get_dt_stamp()
    log = logger.get(
        os.path.join(main_config.log_dir, '%s_%s' % (experiment, dt_string)))
    if num_gpus:
        gpu_device = ['/gpu:%d' % i for i in range(num_gpus)]
    if test and save_test_npz and out_dir is None:
        raise RuntimeError('You must specify an out_dir.')
    if use_db:
        exp_params = db.get_parameters(
            log=log,
            experiment=experiment,
            random=random)[0]
    else:
        exp = py_utils.import_module(experiment, pre_path='experiments')
        exp_params = exp.experiment_params()
        exp_params['_id'] = -1
        exp_params['experiment'] = experiment
        if model is not None:
            exp_params['model'] = model
        else:
            assert len(exp_params['model']) > 1, 'No model name supplied.'
            exp_params['model'] = exp_params['model'][0]
        if train is not None:
            exp_params['train_dataset'] = train
        if val is not None:
            exp_params['val_dataset'] = val
    # if reduction or out_dir is not None or transfer:
    #     fine_tune = get_fine_tune_params(
    #         out_dir=out_dir, reduction=reduction)
    # else:
    #     pass
    results = model_tools.build_model(
        exp_params=exp_params,
        dt_string=dt_string,
        log=log,
        test=test,
        config=main_config,
        use_db=use_db,
        num_batches=num_batches,
        map_out=map_out,
        placeholders=placeholders,
        add_config=add_config,
        gpu_device=gpu_device,
        cpu_device=cpu_device,
        checkpoint=checkpoint)
    if test and save_test_npz:
        # Save results somewhere safe
        py_utils.make_dir(out_dir)
        results['checkpoint'] = checkpoint
        results['model'] = model
        results['experiment'] = experiment
        np.savez(os.path.join(out_dir, results['exp_label']), **results)
        try:
            plotter(fn=os.path.join(out_dir, results['exp_label']), f=os.path.join(out_dir, results['exp_label']), test_dict=results['test_dict'])
        except:
            print("Could not save BSDS figs.")
    log.info('Finished.')


if __name__ == '__main__':
    """Flags are:

    experiment (STR): the name of a Parent experiment class you are using.
        model (STR): the name of a single model you want to
            train/test (overwrites parent experiment params)
        train (STR): a dataset class you want to use for
            training (defaults to tfrecords;
            overwrites parent experiment params)
        val (STR): see above
        num_batches (int): overwrite the experiment default
            # of validation_steps per validation
    ckpt (STR): the full path to a model checkpoint you will
        restore "model" with.
    reduction (int): DEPRECIATED reduce dataset size in training by a factor
    out_dir (STR): custom directory name to store your val output
    gpu (STR): gpu name for scoping
    cpu (STR): cpu name for scoping
    add_config (STR): add a string to your model-specific saved config file
    map_out (STR): only used if mAP is requested in experiment
        custom output folder.
    transfer (BOOL): DEPRECIATED custom transfer learning approach
    placeholders (BOOL): Use placeholders in training/val.
    test (BOOL): Use dataset-class test routine. Saves a npz with test data.
    no_db (BOOL): Do not use database functions.
    no_npz (BOOL): Does not save npz with test data. Only for test data!
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment',
        type=str,
        default=None,
        help='Model script.')
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default=None,
        help='Model script.')
    parser.add_argument(
        '--train',
        dest='train',
        type=str,
        default=None,
        help='Train data.')
    parser.add_argument(
        '--val',
        dest='val',
        type=str,
        default=None,
        help='Validation dataset.')
    parser.add_argument(
        '--num_batches',
        dest='num_batches',
        type=int,
        default=None,
        help='Cap the number of batches to run.')
    parser.add_argument(
        '--ckpt',
        dest='checkpoint',
        type=str,
        default=None,
        help='Path to model ckpt for finetuning.')
    parser.add_argument(
        '--reduction',
        dest='reduction',
        type=int,
        default=None,
        help='Dataset reduction factor.')
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        type=str,
        default=None,
        help='Customized output directory for finetuned model.')
    parser.add_argument(
        '--num_gpus',
        dest='num_gpus',
        type=int,
        default=None,
        help='Number of GPUs to use.')
    parser.add_argument(
        '--cpu',
        dest='cpu_device',
        type=str,
        default='/cpu:0',
        help='CPU device.')
    parser.add_argument(
        '--add_config',
        dest='add_config',
        type=str,
        default=None,
        help='Add this information to the config.')
    parser.add_argument(
        '--map_out',
        dest='map_out',
        type=str,
        default=None,
        help='Folder to put your predictions.')
    parser.add_argument(
        '--transfer',
        dest='transfer',
        action='store_true',
        help='Enable the transfer learning routine.')
    parser.add_argument(
        '--placeholders',
        dest='placeholders',
        action='store_true',
        help='Use placeholders.')
    parser.add_argument(
        '--test',
        dest='test',
        action='store_true',
        help='Test model on data.')
    parser.add_argument(
        '--no_db',
        dest='use_db',
        action='store_false',
        help='Do not use the db.')
    parser.add_argument(
        '--no_npz',
        dest='save_test_npz',
        action='store_false',
        help='Don\'t save an extra test npz.')
    parser.add_argument(
        '--viz_ae',
        dest='viz_ae',
        action='store_true',
        help='Visualize AE.')
    args = parser.parse_args()
    main(**vars(args))

