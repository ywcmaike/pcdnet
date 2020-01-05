from .p2m_dataset import P2MDataSet
from .custom_shapenet import ShapeNetDataSet

class Dataset:
    def __init__(self, data_name, data_root, categories, param, train):
        if data_name.strip() == 'p2m':
            self.dataset = P2MDataSet(data_root, categories=categories , train=train)
        elif data_name.strip() == 'shapenet':
            self.dataset = ShapeNetDataSet(
                data_root, param, categories=categories, train=train)
        else:
            print("only support ['p2m', 'shapenet] for now")

def creare_dataset(dataset_name, data_root, categories, param, train):
    return Dataset(dataset_name, data_root, categories, param, train)

