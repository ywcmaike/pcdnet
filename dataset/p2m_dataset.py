import torch
import pickle
from torch_geometric.data import Dataset, Data
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil


class P2MDataSet(Dataset):

    category_ids = {
        'bench': '02828884',
        'chair': '03001627',
        'lamp': '03636649',
        'speaker': '03691459',
        'firearm': '04090263',
        'table': '04379243',
        'watercraft': '04530566',
        'plane': '02691156',
        'cabinet': '02933112',
        'car': '02958343',
        'monitor': '03211117',
        'couch': '04256520',
        'cellphone': '04401088'
    }

    def __init__(self, root, categories=None, train=True, transform=None, pre_transform=None, pre_filter=None):
        if categories is None:
            categories = list(self.category_ids)
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories

        self.cat_dict = {}
        self.all_files = []
        self.lists = ['train_list.txt', 'test_list.txt']

        if train:
            self.list_path = os.path.join('./dataset/p2m',  self.lists[0])
        else:
            self.list_path = os.path.join('./dataset/p2m',  self.lists[1])

        for idx, cat in enumerate(self.categories):
            self.cat_dict[cat] = []
            cat_id = self.category_ids[cat]
            with open(self.list_path, 'r') as f:
                lines = f.readlines()
                print('reading ', cat, ' in ', self.list_path)
                for line in tqdm(lines):
                    mesh_path = os.path.join(root, line.strip())
                    img_path = mesh_path.replace('.dat', '.png')
                    id = mesh_path.split('/')[-4]
                    if id == cat_id:
                        if os.path.exists(mesh_path) and os.path.exists(img_path):
                            self.cat_dict[cat].append(mesh_path)
                        # else:
                        #     print('data ', mesh_path, ' or ', img_path, ' not exist !!!')

        for cat, file_list in self.cat_dict.items():
            print('collecting ', cat, ' in ', self.list_path)
            for mesh_path in tqdm(file_list):
                splits = mesh_path.split('/')
                pt_file = '{}_{}_{}.pt'.format(
                    cat, splits[-3], splits[-1].split('.')[-2])
                self.all_files.append(pt_file)

        super(P2MDataSet, self).__init__(
            root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.lists

    @property
    def processed_file_names(self):
        return self.all_files

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        img_transformations = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor()])

        for cat, file_list in self.cat_dict.items():
            print('packing ', cat)
            for mesh_path in tqdm(file_list):
                splits = mesh_path.split('/')
                pt_file = os.path.join(self.processed_dir, '{}_{}_{}.pt'.format(
                    cat, splits[-3], splits[-1].split('.')[-2]))
                if os.path.exists(pt_file):
                    continue

                mesh_data = torch.Tensor(pickle.load(
                    open(mesh_path, 'rb'), encoding='latin1'))

                img_path = mesh_path.replace('.dat', '.png')
                img_array = np.array(Image.open(img_path))
                img_array[np.where(img_array[:, :, 3] == 0)] = 255
                img_data = img_transformations(
                    Image.fromarray(img_array[:, :, :3])).unsqueeze(0)

                data = Data(
                    y=img_data, pos=mesh_data[:, :3], norm=mesh_data[:, 3:])

                torch.save(data, pt_file)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])

    def download(self):
        shutil.copy('./dataset/p2m/test_list.txt', self.raw_dir)
        shutil.copy('./dataset/p2m/train_list.txt', self.raw_dir)


if __name__ == '__main__':
    dataset = P2MDataSet('/home/lihai/Downloads/P2MDataSet/')
