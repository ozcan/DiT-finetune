
import glob
import json
import torch
from tqdm import tqdm
from transformers import pipeline

# code below needed if png files are big
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
# ---


device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(task="image-classification", 
                model="./dit-base-finetuned-rvlcdip-custom/checkpoint-23/", device=device)

predictions = {}

for img_path in tqdm(glob.glob("isolated_test/*.png")):
    predictions[img_path] = pipe(img_path)

json.dump(predictions, open("isolated_test.json", "w"), indent=4)

