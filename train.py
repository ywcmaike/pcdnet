import time
import torch
from tqdm import tqdm
from option import create_parser
from dataset import creare_dataset
from logger import create_logger
from model import create_model
from saver import create_saver
from mesh import create_mesh
from loss import compute_loss, compute_fscore
from torch_geometric.data import DataLoader, DataListLoader, Data
import torch_geometric

def make_input_dict(param, data):
    input_dict = {}
    if param.data_type == 'p2m':
        input_dict['render_img'] = data.y
        input_dict['normal_img'] = None
        input_dict['mask'] = None
        input_dict['proj_mat'] = None
        input_dict['batch'] = data.batch
        input_dict['vertex'] = data.pos
        input_dict['face'] = None
        input_dict['normal'] = data.norm
        input_dict['sample'] = None

    elif param.data_type == 'shapenet':
        input_dict['render_img'] = data.render_img
        input_dict['normal_img'] = None
        input_dict['mask'] = None
        input_dict['proj_mat'] = data.proj_mat
        input_dict['batch'] = data.batch
        input_dict['vertex'] = data.pos
        input_dict['face'] = data.face
        input_dict['normal'] = None
        input_dict['sample'] = data.sample_pos
        
    return input_dict

def run_eval(test_loader, test_model, init_mesh, logger, n_epoch, device, param):
    f_score = 0
    n_iter = 0

    print('start evaluating for epoch ', n_epoch)
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)

            input_dict = make_input_dict(param, data)
            output_dict = {}
            points, pre_coords, coords, edges, faces, _, output_img = test_model(input_dict['render_img'], input_dict['proj_mat'], logger, n_iter)
            output_dict['pre_coords'] = pre_coords
            output_dict['points'] = points
            output_dict['coords'] = coords
            output_dict['edges'] = edges
            output_dict['faces'] = faces
            output_dict['output_img'] = output_img
            
            f_score += compute_fscore(input_dict, output_dict, 1e-4)
            n_iter += 1

            if logger is not None:
                if n_iter % param.eval_mesh_freq_iter == 0:
                    logger.save_mesh('eval_mesh3_{}'.format(n_epoch), output_dict['coords'][-1][0], n_iter, output_dict['faces'][-1][0].to(device))
                    logger.save_mesh('eval_gt_pcl', input_dict['sample'][0], n_iter)
                    logger.save_mesh('eval_detail_pcl', output_dict['points'][0], n_iter)

    print('finish evaluating for epoch ', n_epoch)
    if logger is not None:
        logger.add_eval('f score', f_score / test_loader.dataset.__len__(), n_epoch)
    print('{} epoch f1 score = {}'.format(n_epoch, f_score / test_loader.dataset.__len__()))

def train(param):
    # use cuda
    device = torch.device(
        'cuda:{}'.format(param.gpu_ids[0]) if param.use_cuda and torch.cuda.is_available() else 'cpu')
    # dataset
    train_dataset = creare_dataset(
        param.data_type, param.data_root, categories=param.train_category, param=param, train=True)
    test_dataset = creare_dataset(
        param.data_type, param.data_root, categories=param.eval_category, param=param, train=False)
    # dataloader
    train_loader = DataLoader(train_dataset.dataset, batch_size=param.batch_size, shuffle=True, num_workers=param.num_workers, pin_memory=param.pin_mem)
    test_loader = DataLoader(test_dataset.dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.num_workers, pin_memory=param.pin_mem)
    # logger
    logger = None
    if param.use_logger:
        logger = create_logger(param.log_path + '_' + param.name)
    # predefined mesh
    host_init_mesh = create_mesh(param)
    device_init_mesh = create_mesh(param, device)
    # model
    model = create_model(host_init_mesh, param, train=True)
    # checkpoint
    saver = create_saver(param.save_path + '_' + param.name)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
    # iteration
    n_iter = 0
    # parallel setting
    if param.muti_gpu:
        model = torch.nn.DataParallel(model, param.gpu_ids)
    model = model.to(device)

    for epoch in range(param.epoch):
        print('start traning for epoch ', epoch)
        for data in tqdm(train_loader):

            # read data
            data = data.to(device)
            optimizer.zero_grad()

            # collect input
            input_dict = make_input_dict(param, data)
            output_dict = {}

            points, pre_coords, coords, edges, faces, vmasks, output_img = model(input_dict['render_img'], input_dict['proj_mat'], logger, n_iter)
            output_dict['pre_coords'] = pre_coords
            output_dict['points'] = points
            output_dict['coords'] = coords
            output_dict['edges'] = edges
            output_dict['faces'] = faces
            output_dict['vmasks'] = vmasks
            output_dict['output_img'] = output_img

            total_loss, loss_dict, gt_sample_pos = compute_loss(param, device_init_mesh, input_dict, output_dict)

            total_loss.backward()
            optimizer.step()

            n_iter += 1

            if logger is not None:
                if n_iter % param.loss_freq_iter == 0:
                    logger.add_loss('total loss', total_loss.item(), n_iter)
                    logger.add_losses('losses', loss_dict, n_iter)

                if n_iter % param.check_freq_iter == 0:
                    logger.add_gradcheck(model.named_parameters(), n_iter)
                    logger.add_image('input image', input_dict['render_img'], n_iter)
                    
                    # logger.add_projectionU('gt_project', gt_sample_pos, input_dict['proj_mat'] n_iter)
                    if input_dict['proj_mat'] is not None:
                        logger.add_projectionU('gt_project', input_dict['sample'][0].unsqueeze(0), input_dict['proj_mat'], n_iter)
                        for i in range(len(output_dict['coords'])):
                            logger.add_projectionU('project{}'.format(i), output_dict['coords'][0], input_dict['proj_mat'], n_iter)

                    else:
                        logger.add_projection('gt_project', input_dict['sample'][0].unsqueeze(0), n_iter)
                        for i in range(len(output_dict['coords'])):
                            logger.add_projection('project{}'.format(i), output_dict['coords'][0], n_iter)

                if n_iter % param.train_mesh_freq_iter == 0:
                    for i in range(len(output_dict['coords'])):
                        logger.save_mesh('mesh{}'.format(i),  output_dict['pre_coords'][i][0], n_iter, output_dict['faces'][i][0])
                        logger.save_mesh('mesh{}'.format(i),  output_dict['coords'][i][0], n_iter, output_dict['faces'][i][0])
                        
                    logger.save_mesh('gt_pcl', input_dict['sample'][0], n_iter)
                    logger.save_mesh('detail_pcl', output_dict['points'][0], n_iter)

        print('finish traning for epoch ', epoch)
        if isinstance(model, torch.nn.DataParallel):
            saver.save_model(model.module.state_dict(), epoch)
        else:
            saver.save_model(model.state_dict(), epoch)
        # saver.save_optimizer(optimizer.state_dict(), epoch)

        print('finish saving checkpoint for epoch ', epoch)
        if n_iter % param.eval_freq_epoch == 0:
            run_eval(
                test_loader, model, host_init_mesh, logger, epoch, device, param)


if __name__ == '__main__':
    # parameters
    param = create_parser()
    # start training
    train(param)
