import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import config

class Dataset(data.Dataset):
    def __init__(self, data_list=[], skip_frame=1, time_step=30):
        '''
        UCF101
        '''
        #one-hot
        self.labels = []
        self.images = []
        self.use_mem = False

        self.skip_frame = skip_frame
        self.time_step = time_step
        self.data_list = self._build_data_list(data_list)

    def __len__(self):
        return len(self.data_list) // self.time_step

    def __getitem__(self, index):
        #time_step
        index = index * self.time_step
        imgs = self.data_list[index:index + self.time_step]

        if self.use_mem:
            X = [self.images[x[3]] for x in imgs]
        else:
            X = [self._read_img_and_transform(x[2]) for x in imgs]

        #tensor
        X = torch.stack(X, dim=0)
        y = torch.tensor(self._label_category(imgs[0][0]))
        return X, y

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((config.img_w, config.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img:str):
        return self.transform(Image.open(img).convert('RGB'))

    def _build_data_list(self, data_list=[]):

        if len(data_list) == 0:
            return []

        data_group = {}
        for x in tqdm(data_list, desc='Building dataset'):
            #classnam,videoname
            [classname, videoname] = x[0:2]
            if classname not in data_group:
                data_group[classname] = {}
            if videoname not in data_group[classname]:
                data_group[classname][videoname] = []

            if self.use_mem:
                self.images.append(self._read_img_and_transform(x[2]))

            data_group[classname][videoname].append(list(x) + [len(self.images) - 1])

        self.labels = list(data_group.keys())

        ret_list = []
        n = 0
        for classname in data_group:
            video_group = data_group[classname]
            for videoname in video_group:
                video_pad_count = len(video_group[videoname]) % self.time_step
                video_group[videoname] += [video_group[videoname][-1]] * (self.time_step - video_pad_count)
                ret_list += video_group[videoname]
                n += len(video_group[videoname])

        return ret_list

    def _label_one_hot(self, label):
        if label not in self.labels:
            raise RuntimeError('label！')
        one_hot = [0] * len(self.labels)
        one_hot[self.labels.index(label)] = 1
        return one_hot

    def _label_category(self, label):
  
        if label not in self.labels:
            raise RuntimeError('label！')
        c_label = self.labels.index(label)
        return c_label
