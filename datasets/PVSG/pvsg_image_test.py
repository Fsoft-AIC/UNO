import os, glob, json
from pathlib import Path

from torch.utils.data import Dataset

from datasets.PVSG.utils import PVSGAnnotation

from datasets.PVSG.pipelines.loading import LoadAnnotationsDirect
from datasets.PVSG.pipelines.transforms_single import make_transforms


class PVSGImageDataset(Dataset):
    def __init__(self,
                 data_root='./data/',
                 annotation_file='pvsg_train_new.json',
                 classes_file='pvsg_classes.json',
                 split='train',
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

        # collect class names
        self.RELATION_CLASSES = classes['relations']
        self.BACKGROUND_CLASSES = ['background']
        self.CLASSES = classes['objects']
        self.num_classes = len(self.CLASSES)  # 126
        self.num_relation_classes = len(self.RELATION_CLASSES)
        self.videos = anno['videos']
        self.images = anno['images']
        self.annotation_loader = LoadAnnotationsDirect(self.cates2id, self.num_classes, self.num_relation_classes)

    def cates2id(self, category):
        class2ids = dict(
            zip(self.CLASSES + self.BACKGROUND_CLASSES,
                range(len(self.CLASSES + self.BACKGROUND_CLASSES))))
        return class2ids[category]

    def __getitem__(self, idx):
        image_results = self.images[idx]
        video_results = self.videos[image_results['video_id']]
        img, targets = self.annotation_loader(self.data_root, video_results, image_results)
        img, targets = self._transforms(img, targets)
        return img, targets

    def __len__(self):
        return len(self.images)
    
def build(image_set, args):
    root = Path('/your_dir')
    assert root.exists(), f'provided PVSG path {root} does not exist'
   
    img_folder = root
    anno_file = f'pvsg_{image_set}_new_relations_1.json'
    classes_file = f'pvsg_classes.json'
    
    dataset = PVSGImageDataset(img_folder, anno_file, classes_file, image_set, make_transforms(image_set, 448))

    return dataset
