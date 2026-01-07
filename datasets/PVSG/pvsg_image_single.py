import os, glob, json
from pathlib import Path

from torch.utils.data import Dataset

from datasets.PVSG.pipelines.loading import LoadAnnotationsDirect
from datasets.PVSG.pipelines.transforms_single import make_transforms
import numpy as np
import copy 

class PVSGImageSingleDataset(Dataset):
    def __init__(self,
                 data_root='./data/',
                 annotation_file='pvsg_train_new.json',
                 classes_file='pvsg_classes.json',
                 split='train',
                 video_name='',
                 transforms=None,):
        assert data_root is not None
        
        print('Loading annotations')
        self._transforms = transforms
        self.data_root = Path(data_root)
        anno_file = data_root / annotation_file
        classes_all_file = data_root / classes_file

        with open(classes_all_file, 'r') as f:
            classes = json.load(f)

        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.INSTANCE_OFFSET = 1000

        # collect class names        
        self.THING_CLASSES = classes['objects']['thing']  # 115
        self.STUFF_CLASSES = classes['objects']['stuff']  # 11
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.num_thing_classes = len(self.THING_CLASSES)
        self.num_stuff_classes = len(self.STUFF_CLASSES)
        self.num_classes = len(self.CLASSES)  # 126
        self.BACKGROUND_CLASSES = ['background']
        
        self.RELATION_CLASSES = classes['relations']
        self.num_relation_classes = len(self.RELATION_CLASSES)
        self.video_name = video_name
        print(f"{split} video num: {len(anno['videos'])}")
        print(f"{split} image num: {len(anno['images'])}")
        self.videos = anno['videos'][video_name]
        single_vid = []
        for frame in anno['images']:
            if frame['video_id'] == video_name:
                single_vid.append(frame)
        self.images = single_vid
        self.annotation_loader = LoadAnnotationsDirect(self.cates2id, self.num_classes, self.num_relation_classes)

        print('Finished loading annotations')

    def cates2id(self, category):
        class2ids = dict(
            zip(self.CLASSES + self.BACKGROUND_CLASSES,
                range(len(self.CLASSES + self.BACKGROUND_CLASSES))))
        return class2ids[category]

    def __getitem__(self, idx):
        image_results = self.images[idx]
        video_results = self.videos
        img, targets = self.annotation_loader(self.data_root, video_results, image_results)
        targets['video_id'] = self.video_name
        targets['frame_id'] = self.images[idx]['frame_id']
        img, targets = self._transforms(img, targets)
        
        return img, targets

    def __len__(self):
        return len(self.images)
    

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided PVSG path {root} does not exist'
   
    img_folder = root
    anno_file = f'pvsg_{image_set}_new.json'
    classes_file = f'pvsg_classes.json'
    video_name = '0001_4164158586' 
    dataset = PVSGImageSingleDataset(img_folder, anno_file, classes_file, image_set, video_name, make_transforms(image_set, args.image_size))

    return dataset
