"""
Usecase 2 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --config=CONFIG    Specify run config to use [default: config.yml]
"""
import sys
from datetime import datetime
from pathlib import Path
import shutil

import torch
import torchsummary
from tqdm import tqdm

from deep_learning import Trainer
from deep_learning.models import get_model
from deep_learning.loss_functions import get_loss
from deep_learning.utils import showexample
from data_loading import get_loader, get_filtered_loader, get_batch

import re

from docopt import docopt
import yaml


def scoped_get(key, *scopestack):
    for scope in scopestack:
        value = scope.get(key)
        if value is not None:
            return value
    raise ValueError(f'Could not find "{key}" in any scope.')


def get_dataloader(name):
    if name in dataset_cache:
        return dataset_cache[name]
    if name in config['datasets']:
        ds_config = config['datasets'][name]
        if 'batch_size' not in ds_config:
            ds_config['batch_size'] = config['batch_size']
        dataset_cache[name] = get_loader(**ds_config)
        return dataset_cache[name]
    else:
        func, arg = re.search(r'(\w+)\((\w+)\)', name).groups()
        if func == 'slump_tiles':
            ds_config = config['datasets'][arg]
            if 'batch_size' not in ds_config:
                ds_config['batch_size'] = config['batch_size']
            dataset_cache[name] = get_filtered_loader(**ds_config)
    return dataset_cache[name]



if __name__ == "__main__":
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config_file = Path(cli_args['--config'])
    config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)

    modelclass = get_model(config['model'])
    model = modelclass(config['input_channels'], 1, base_channels=config['modelscale'])

    # TODO: Resume from checkpoint

    trainer = Trainer(model)

    lr = config['learning_rate']
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(trainer.model, [(config['input_channels'], 256, 256)])
        sys.exit(0)

    dataset_cache = {}

    vis_batch = get_batch(config['visualization_tiles'])
    vis_imgs = vis_batch[0].to(trainer.dev)

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)

    shutil.copy(config_file, log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    for phase in config['schedule']:
        print(f'Starting phase "{phase["phase"]}"')
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(f'Phase {phase["phase"]}', file=f)
        for epoch in range(phase['epochs']):
            # Epoch setup
            loss_fn = get_loss(scoped_get('loss_function', phase, config))
            trainer.loss_function = loss_fn.to(trainer.dev)

            datasets_config = scoped_get('datasets', phase, config)

            for step in phase['steps']:
                if type(step) is dict:
                    assert len(step) == 1
                    (command, key), = step.items()
                else:
                    command = step
                if command == 'train_on':
                    # Training step
                    data_loader = get_dataloader(key)
                    trainer.train_epoch(tqdm(data_loader))
                    metrics = trainer.metrics.evaluate()
                    logstr = f'Epoch {trainer.epoch:02d} - Train: ' \
                           + ', '.join(f'{key}: {val:.2f}' for key, val in metrics.items())
                    print(logstr)
                    with (log_dir / 'metrics.txt').open('a+') as f:
                        print(logstr, file=f)
                    # Save model Checkpoint
                    torch.save(trainer.model.state_dict(), checkpoints / f'{trainer.epoch:02d}.pt')
                elif command == 'validate_on':
                    # Validation step
                    data_loader = get_dataloader(key)
                    trainer.val_epoch(data_loader)
                    metrics = trainer.metrics.evaluate()
                    logstr = f'Epoch {trainer.epoch:02d} - Val: ' \
                           + ', '.join(f'{key}: {val:.2f}' for key, val in metrics.items())
                    print(logstr)
                    with (log_dir / 'metrics.txt').open('a+') as f:
                        print(logstr, file=f)
                elif command == 'log_images':
                    with torch.no_grad():
                        pred = trainer.model(vis_imgs)
                    for i, tile in enumerate(config['visualization_tiles']):
                        filename = log_dir / tile / f'{trainer.epoch}.png'
                        filename.parent.mkdir(exist_ok=True)
                        showexample(vis_batch, pred, i, filename)
