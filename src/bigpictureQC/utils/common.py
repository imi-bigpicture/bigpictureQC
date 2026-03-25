import os 
import random 
import torch 
import numpy as np

def seed_everything(seed=1234):
    """Set seed for multiple random processes."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_count(batch, num_classes):
    """Count the unique classes present in a batch."""

    batch = torch.argmax(batch, dim=1)
    device = batch.get_device() if batch.is_cuda else 'cpu'
    targets = torch.zeros(size=(batch.size()[0], num_classes), device=device)

    for idx in range(batch.size()[0]):
        mask = batch[idx]
        unique = torch.unique(mask)

        for class_value in range(num_classes):
            if class_value in unique:
                targets[idx, class_value] = 1.0

    return targets


