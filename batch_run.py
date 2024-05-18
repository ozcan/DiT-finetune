from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

pngs = [f for f in glob.glob("images/*.png")]

class ListDataset(Dataset):
    def __init__(self):
        self.original_list = pngs

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

dataset = ListDataset()
dataloader = DataLoader(dataset, batch_size=256)
pipe = pipeline(task="image-classification", model="microsoft/dit-base-finetuned-rvlcdip", device=0)

with open("predictions.txt", "w") as out:
    for batch in tqdm(dataloader):
        outputs = pipe(batch)
        for image_file, output in zip(batch, outputs):
           out.write(f"Filename: {image_file}, Output: {output}")

