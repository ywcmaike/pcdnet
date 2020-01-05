import time
import torch
from tqdm import tqdm
from option import create_parser
from dataset import creare_dataset
from model import create_model
from mesh import create_mesh
from loss import compute_fscore
from torch_geometric.data import DataLoader

category_f_score = {
    'bench': [],
    'chair': [],
    'lamp': [],
    'speaker': [],
    'firearm': [],
    'table': [],
    'watercraft': [],
    'plane': [],
    'cabinet': [],
    'car': [],
    'monitor': [],
    'couch': [],
    'cellphone': []
}

if __name__ == '__main__':
    # parameters
    param = create_parser()
    # use cuda
    device = torch.device(
        'cuda:{}'.format(param.gpu_ids[0]) if param.use_cuda and torch.cuda.is_available() else 'cpu')
    # predefined mesh
    init_mesh = create_mesh(param.obj_file, device=device)
    # model
    test_model = create_model(
        device, init_mesh, train=True, checkpoint='/home/lihai/workspace/eval/pyg_model/save/model_epoch19_2019_08_30_20_00_25.pth')

    # if param.muti_gpu:
    #     model = torch.nn.DataParallel(model, param.gpu_ids)

    f_total = 0
    total_count = 0

    for cat in category_f_score.keys():
        # dataset
        test_dataset = creare_dataset(
            param.dataset, categories=cat, train=False)
        # dataloader
        test_loader = DataLoader(test_dataset.dataset, batch_size=param.batch_size,
                                shuffle=False, num_workers=param.num_workers, pin_memory=param.pin_mem)

        f_score = 0
        with torch.no_grad():
            for data in tqdm(test_loader):
                data = data.to(device)
                img = data.y

                coords = test_model(img)
                f_score += compute_fscore(coords[0], data, 1e-4)

        f_total += f_score
        total_count += test_loader.dataset.__len__()
        category_f_score[cat].append(f_score / test_loader.dataset.__len__())

    print(category_f_score)
    print('total avg f score ', f_total / total_count)
