import importlib


def load_data(dataset_config: dict, dataloader_config: dict):
    """Return dataloaders from dataset_config"""
    try:
        dataset_module = importlib.import_module(f'dataloaders.{dataset_config["name"]}')
    except Exception as e:
        print(f'Error: {dataset_config}')
        raise e
    _load_data = getattr(dataset_module, 'load_data')
    return _load_data(dataset_config, dataloader_config)
