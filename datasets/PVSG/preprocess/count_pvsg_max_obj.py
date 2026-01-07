import json
import os
import glob 

from pathlib import Path




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


THING_CLASSES = anno['objects']['thing']  # 115
STUFF_CLASSES = anno['objects']['stuff']  # 11
BACKGROUND_CLASSES = ['background']
CLASSES = THING_CLASSES + STUFF_CLASSES
num_thing_classes = len(THING_CLASSES)
num_stuff_classes = len(STUFF_CLASSES)
num_classes = len(CLASSES)  # 126
relations = anno['relations']

num_vid_no_frame = 0
num_vid_frame = 0

video_data = {}
max_rel = 0
total_rel = 0
for video_anno in anno['data']:
    total_rel += len(video_anno['relations'])
    max_rel = max(max_rel, len(video_anno['relations']))
print(total_rel)
print(max_rel)
# ignore_list = ['1014_5820886415', 'a383d099-5eef-48b5-9d1b-5e2d97632725']

# max_obj = 0
# video_ids, img_names = [], []
# for split in ['train', 'val']:
#     save_file_name = f'./annotations/PVSG/pvsg_{split}_new.json'
#     save_file_cm_name = f'{data_root}/pvsg_{split}_new.json'

#     for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
#         for video_id in anno['split'][data_source][split]:
#             if video_id in ignore_list:
#                 continue
#             video_ids.append(video_id)
#             frame_name = glob.glob(
#                 os.path.join(data_root, data_source, 'frames', video_id,
#                                 '*.png'))
#             img_names += frame_name
#             # if len(frame_name) > 0:
#             max_obj = max(max_obj, len(video_data[video_id]['objects']))
    # with open(save_file_name, "w") as f:
    #     json.dump(data_dict, f,indent=2)
        
    # with open(save_file_cm_name, "w") as f:
    #     json.dump(data_dict, f,indent=2)


# images = []
# for itm in img_names:
#     vid = itm.split(sep='/')[-2]
#     vid_anno = video_data[vid]
#     print(itm)
#     img_path = itm,
#     ann_path = itm.replace('frames', 'masks'),
#     objects = vid_anno['objects'],

#     assert os.path.exists(
#         img_path), f"File not found: {split, img_path}"
#     assert os.path.exists(
#         ann_path), f"File not found: {split, ann_path}"
