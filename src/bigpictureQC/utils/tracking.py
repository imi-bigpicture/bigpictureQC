import wandb
from typing import Optional, List


def initialize_wandb(
    cfg,
    tags: Optional[List] = None,
    key: Optional[str] = "",
    fold = 0
):
    command = f"wandb login {key}"
    if tags == None:
        tags = []

    run = wandb.init(
        settings=wandb.Settings(start_method='fork'),
        project=cfg.wandb_config.get('project'),
        entity=cfg.wandb_config.get('username'),
        name=cfg.wandb_config.get('exp_name') + '_fold_{}'.format(fold),
        dir=cfg.wandb_config.get('dir'),
        tags=tags,
        config=dict(cfg.settings),
    )
    config_paths = [cfg.batch_iterator_config.get('user_config_path'), cfg.batch_iterator_config.get('val_config_path')]
    upload_config_files(run, config_paths)
    return run


def upload_config_files(run, config_paths):
    for path in config_paths:
        run.save(path)