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

fail_frame = 0

THING_CLASSES = anno['objects']['thing']  # 115
STUFF_CLASSES = anno['objects']['stuff']  # 11
BACKGROUND_CLASSES = ['background']
CLASSES = THING_CLASSES + STUFF_CLASSES
num_thing_classes = len(THING_CLASSES)
num_stuff_classes = len(STUFF_CLASSES)
num_classes = len(CLASSES)  # 126
relations_classes = anno['relations']

class2ids = dict(
            zip(CLASSES + BACKGROUND_CLASSES,
                range(len(CLASSES + BACKGROUND_CLASSES))))

relation2ids = dict(zip(relations_classes, range(len(relations_classes))))

fail_frame_list = []
num_vid_no_frame = 0
num_vid_frame = 0
num_relation_frame_skip = 0
video_data = {}
for video_anno in anno['data']:
    video_data[video_anno['video_id']] = video_anno

fail_r = []
for split in ['train', 'val']:
    save_file_name = f'./annotations/PVSG/pvsg_{split}_new_relations.json'
    save_file_cm_name = f'{data_root}/pvsg_{split}_new.json'

    for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
        for video_id in anno['split'][data_source][split]:
            
            object2id = dict()
            for obj in video_data[video_id]['objects']:
                object2id[obj['object_id']] = obj['category']
            
            frame_name = glob.glob(
                os.path.join(data_root, data_source, 'frames', video_id,
                                '*.png'))
            frame_id = sorted([fid.split('/')[-1].split('.')[0] for fid in frame_name])
            temp = []
            for frame in frame_id:
                relations = []
                subjects_list = []
                objects_list = []
                classes = []
                instance_ids = []

                objects_info = video_data[video_id]['objects']

                img_path = os.path.join(data_source, 'frames', video_id, f'{frame}.png')
                
                mask_path = os.path.join(data_root, img_path.replace('frames', 'masks'))
        
                pan_mask = np.array(Image.open(mask_path).convert('P')).astype(np.int64)  # palette format saved one-channel image
                if (max(np.unique(pan_mask))-1) >= len(objects_info):
                    fail_frame += 1
                    fail_frame_list.append(os.path.join(video_id, f'{frame}.png'))
                    continue
                
                # filter background (void) class frame

                for instance_id in np.unique(pan_mask):  # 0,1...n object id
                    if instance_id == 0:  # no segmentation area
                        continue
                    else:  # gt_label & gt_masks do not include "void"
                        instance_ids.append(instance_id - 1)
                        category = objects_info[instance_id - 1]['category']
                        semantic_id = class2ids[category]
                        classes.append(semantic_id)
                if len(classes) == 0:
                    fail_frame += 1
                    fail_frame_list.append(os.path.join(video_id, f'{frame}.png'))
                    continue
                
                for relation_dict in video_data[video_id]['relations']:
                    subject_id = relation_dict[0]
                    object_id = relation_dict[1]
                    relationship = relation_dict[2]
                    if relationship not in relation2ids:
                        if relationship not in fail_r:
                            print(relationship)
                            fail_r.append(relationship)
                            
                        num_relation_frame_skip += 1
                        continue
                    time_range = relation_dict[3]
                    for time in time_range:
                        if time[0] <= int(frame) <= time[1]:
                            if subject_id not in instance_ids or object_id not in instance_ids:
                                continue
                            relations.append({
                                'sub_name': object2id[subject_id],
                                'sub': class2ids[object2id[subject_id]],
                                'sub_id': subject_id,
                                'obj_name': object2id[object_id],
                                'obj': class2ids[object2id[object_id]],
                                'obj_id': object_id,
                                'rel': relationship,
                                'rel_id': relation2ids[relationship],
                            })
                    
                    if subject_id not in subjects_list:
                        subjects_list.append(subject_id)
                            
                    if object_id not in objects_list:
                        objects_list.append(object_id)
                
                if len(relations) == 0:
                    fail_frame += 1
                    fail_frame_list.append(os.path.join(video_id, f'{frame}.png'))
                    continue
                       
                temp.append({
                'id': id_frames,
                'frame_id': int(frame),
                'video_id': video_id,
                'data_source': data_source,
                'img_path': img_path,
                'ann_path': img_path.replace('frames', 'masks'),
                'relations': relations,
                'objects_list': objects_list,
                'subjects_list': subjects_list
                })
                id_frames += 1
            
            data_dict['images'].extend(temp)
            data_dict['videos'][video_id] = {
                'id': id_videos,
                'video_id': video_id,
                'data_source': data_source,
                'objects': video_data[video_id]['objects'],
                'relations': video_data[video_id]['relations'],
            }
            
            id_videos += 1

    print(f"fail relation {num_relation_frame_skip}")
    print(f"fail frame {fail_frame}")

    with open(save_file_name, "w") as f:
        json.dump(data_dict, f,indent=2)
        
    with open(save_file_cm_name, "w") as f:
        json.dump(data_dict, f,indent=2)

