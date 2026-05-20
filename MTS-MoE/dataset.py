import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class ConditionalResizeOrCrop:
    """
    Custom transform that resizes images to 224x224 if they are smaller,
    otherwise center crops them to 224x224.
    """
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.resize = transforms.Resize((target_size, target_size))
        self.crop = transforms.CenterCrop(target_size)

    def __call__(self, img):
        width, height = img.size
        if width < self.target_size or height < self.target_size:
            return self.resize(img)
        else:
            return self.crop(img)


# Transform for images: resize if smaller than 224x224, otherwise crop
transform = transforms.Compose([
    ConditionalResizeOrCrop(224),
    transforms.ToTensor()
])


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Get dataloaders for train and validation sets.

    Expected directory structure:
        data_dir/
            train/
                0_real/
                    img1.jpg
                    ...
                1_fake/
                    img1.jpg
                    ...
            val/
                0_real/
                1_fake/

    Args:
        data_dir: Root directory containing train/val folders
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train dataset: {len(train_dataset)} images, classes: {train_dataset.classes}")
    print(f"Val dataset:   {len(val_dataset)} images, classes: {val_dataset.classes}")

    return train_loader, val_loader


def get_test_dataloader(data_dir, batch_size=32, num_workers=4):
    """
    Get only test dataloader (for evaluation only).

    Args:
        data_dir: Root directory containing test folder
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading

    Returns:
        test_loader
    """
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader