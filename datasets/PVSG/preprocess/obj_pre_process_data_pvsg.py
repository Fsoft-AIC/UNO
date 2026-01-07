import json
import os
import glob 

from pathlib import Path


import numpy as np
from PIL import Image

#fail relation
# getting down on
# squatting on
# climbing
# enclosing
# pouring
# squeezing
# moving
# fail relation 10294
# going down
# fail relation 16082

# ag_file = ag_data_root / 'ag_train_coco_style.json'

# split = 'val'
data_root = Path('/your_dir')
anno_file = data_root / "pvsg.json"

assert anno_file.exists()
assert data_root.exists()

with open(anno_file, 'r') as f:
    anno = json.load(f)

data_dict = dict()
data_dict['images'] = []
data_dict['videos'] = {}

id_videos = 1
id_frames = 1

fail_video = 0

THING_CLASSES = anno['objects']['thing']  # 115
STUFF_CLASSES = anno['objects']['stuff']  # 11
BACKGROUND_CLASSES = ['background']
CLASSES = THING_CLASSES + STUFF_CLASSES
num_thing_classes = len(THING_CLASSES)
num_stuff_classes = len(STUFF_CLASSES)
num_classes = len(CLASSES)  # 126
save_file_name = f'./annotations/PVSG/pvsg_classes.json'
save_file_cm_name = f'{data_root}/pvsg_classes.json'

data_dict = dict()

data_dict['objects'] = anno['objects']

data_dict['all_objects'] = CLASSES
data_dict['relations'] = anno['relations']

with open(save_file_name, "w") as f:
    json.dump(data_dict, f,indent=2)
    
with open(save_file_cm_name, "w") as f:
    json.dump(data_dict, f,indent=2)

