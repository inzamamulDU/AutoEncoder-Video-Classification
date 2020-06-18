import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

default_output_dir = os.path.dirname(os.path.abspath(__file__))
default_src_dir = os.path.join(default_output_dir, 'UCF')
default_test_size = 0.2

def split(src_dir=default_src_dir, output_dir=default_src_dir, size=default_test_size):
    src_dir = default_src_dir if src_dir is None else src_dir
    output_dir = default_output_dir if output_dir is None else output_dir
    size = default_test_size if size is None else size

    for folder in ['train', 'test']:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print('Folder {} is created'.format(folder_path))

    train_set = []
    test_set = []
    classes = os.listdir(src_dir)
    num_classes = len(classes)
    for class_index, classname in enumerate(classes):
        videos = os.listdir(os.path.join(src_dir, classname))
        np.random.shuffle(videos)
        split_size = int(len(videos) * size)

        for i in range(2):
            part = ['train', 'test'][i]
            class_dir = os.path.join(output_dir, part, classname)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

        for i in tqdm(range(len(videos)), desc='[%d/%d]%s' % (class_index + 1, num_classes, classname)):
            video_path = os.path.join(src_dir, classname, videos[i])
            video_fd = cv2.VideoCapture(video_path)

            if not video_fd.isOpened():
                print('Skpped: {}'.format(video_path))
                continue

            video_type = 'test' if i <= split_size else 'train'

            frame_index = 0
            success, frame = video_fd.read()
            video_name = videos[i].rsplit('.')[0]
            while success:
                img_path = os.path.join(output_dir, video_type, classname, '%s_%d.jpg' % (video_name, frame_index))
                cv2.imwrite(img_path, frame)
                info = [classname, video_name, img_path]
                if video_type == 'test':
                    test_set.append(info)
                else:
                    train_set.append(info)
                frame_index += 1
                success, frame = video_fd.read()

            video_fd.release()

        datas = [train_set, test_set]
        names = ['train', 'test']
        for i in range(2):
            with open(output_dir + '/' + names[i] + '.csv', 'w') as f:
                f.write('\n'.join([','.join(line) for line in datas[i]]))

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 make_train_test.py -i path/to/UCF -o path/to/output -s 0.3')
    parser.add_argument('-i', '--src_dir', help='path to UCF datasets', default=default_src_dir)
    parser.add_argument('-o', '--output_dir', help='path to output', default=default_output_dir)
    parser.add_argument('-s', '--size', help='ratio of test sets', default=default_test_size)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    split(**vars(args))
