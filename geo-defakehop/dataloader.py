import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class DeepfakeSatelliteImageDataset(Dataset):
    """Deepfake satellite image dataset."""

    def __init__(self, csv_file, mode, transform=transforms.ToTensor()):
        """
        Args:
            csv_file (string): Path to the csv file.
            mode (string): "train", "test" or "val".
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(csv_file)
        self.images = self.images[self.images["SET"] == mode]
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images.iloc[idx]["FP"])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.images.iloc[idx]["isFake"])

if __name__ == "__main__":
    csv_file = "../split/data_10_10_80.csv"
    image_datasets = {x: DeepfakeSatelliteImageDataset(csv_file=csv_file, mode=x)
                    for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=len(image_datasets[x]), shuffle=True, num_workers=16)
                for x in ['train', 'val', 'test']}

    inputs, classes = next(iter(dataloaders['train']))
    print(inputs.shape)