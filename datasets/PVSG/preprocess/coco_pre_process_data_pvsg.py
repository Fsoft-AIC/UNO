import json
import os
import glob 

from pathlib import Path



import cv2



# ag_file = ag_data_root / 'ag_train_coco_style.json'

split = 'val'
data_root = Path('/your_dir')
anno_file = data_root / "pvsg.json"
save_file_name = f'./annotations/PVSG/pvsg_classes.json'

assert anno_file.exists()
assert data_root.exists()

with open(anno_file, 'r') as f:
    anno = json.load(f)

data_dict = dict()
# data_dict['videos'] = []
# data_dict['images'] = []
# data_dict['annotations'] = []
# data_dict['categories'] = []
# data_dict['rel_categories'] = []

id_videos = 1
id_frames = 1
id_cat = 1
id_rel_cat = 1

THING_CLASSES = anno['objects']['thing']  # 115
STUFF_CLASSES = anno['objects']['stuff']  # 11
BACKGROUND_CLASSES = ['background']
CLASSES = THING_CLASSES + STUFF_CLASSES
num_thing_classes = len(THING_CLASSES)
num_stuff_classes = len(STUFF_CLASSES)
num_classes = len(CLASSES)  # 126
data_dict['objects'] = CLASSES
data_dict['relations'] = anno['relations']
# for class_name in CLASSES:
#     data_dict['categories'].append({
#         'id': id_cat,
#         'name': class_name,
#         'encode_name': None,
#     })
#     id_cat += 1


# for relation in relations:
#     data_dict['rel_categories'].append({
#         'id': id_rel_cat,
#         'name': relation,
#         'encode_name': None,
#     })
#     id_rel_cat += 1


# num_vid_no_frame = 0
# num_vid_frame = 0

# video_ids, img_names = [], []
# for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
#     for video_id in anno['split'][data_source][split]:
#         video_ids.append(video_id)
#         frame_name = glob.glob(
#             os.path.join(data_root, data_source, 'frames', video_id,
#                             '*.png'))
#         img_names += frame_name
#         # if len(frame_name) > 0:
#         frame_id = sorted([fid.split('/')[-1].split('.')[0] for fid in frame_name])
#         # print(frame_id)
#         data_dict['videos'].append({
#             'id': id_videos,
#             'name':f'{data_source}/frames/{video_id}',
#             "vid_train_frames": frame_id,
#         })
#         for frame in frame_id:
#             img_path = os.path.join(data_root, data_source, 'frames', video_id, f'{frame}.png')
#             # print(img_path)
#             img = cv2.imread(img_path)
#             height, width, channels = img.shape
#             data_dict['images'].append({
#             'file_name':f'{data_source}/frames/{video_id}/{frame}.png',
#             'id': id_frames,
#             'frame_id': int(frame),
#             'video_id': id_videos,
#             'width': width,
#             'height': height,
#             "is_vid_train_frame": True,
#             })
#             id_frames += 1
  
#         num_vid_frame += len(frame_name)
      
#         id_videos += 1

with open(save_file_name, "w") as f:
	json.dump(data_dict, f,indent=2)
    

# video_data = {}
# for video_anno in anno['data']:
#     if video_anno['video_id'] in video_ids:
#         video_data[video_anno['video_id']] = video_anno



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
