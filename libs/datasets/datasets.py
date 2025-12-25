import os
import importlib
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator


def load_external_dataset(class_path):
    """
    Load a dataset class from an external module.

    Args:
        class_path: String in format "module.path:ClassName"
                   e.g., "snapformer.training.snap_dataset:SnapDataset"

    Returns:
        The dataset class
    """
    module_path, class_name = class_path.rsplit(':', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_dataset(name, is_training, split, **kwargs):
    """
    A simple dataset builder.

    Supports external datasets via custom_class kwarg:
        custom_class: "module.path:ClassName"
    """
    # Check for external dataset class
    custom_class = kwargs.pop('custom_class', None)
    if custom_class:
        dataset_cls = load_external_dataset(custom_class)
        # Register it for future use
        if name not in datasets:
            datasets[name] = dataset_cls
    elif name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Use custom_class to load external datasets.")
    else:
        dataset_cls = datasets[name]

    dataset = dataset_cls(is_training, split, **kwargs)
    return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
