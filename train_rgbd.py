import os
import time

import torchvision
from torch.nn.parallel.data_parallel import DataParallel

from dataloader.dataset_STB import STB
from model.model import RGBmodel,RGBDmodel
from config import opt
from util.eval_utils import eval_auc

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import cv2
import shutil
import logging
import numpy as np

import torch.nn
from tqdm import tqdm
import random

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR

from tensorboardX import SummaryWriter

from dataloader import loader
from util.generateFeature import GFM

from model.loss import SmoothL1Loss
from util import vis_tool
import matplotlib.pyplot as plt
import json

import time 

JOINT = {
    'STB': 21,
    'nyu':14,
    'dexycb':21,
    'ho3d':21,
    'aloha':21,
    'glove':12 
}


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)  # Set seed for Python's built-in random module
    np.random.seed(seed)  # Set seed for numpy's random module
    torch.manual_seed(seed)  # Set seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch (CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs in PyTorch


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train the model with different configurations')
    parser.add_argument('--root_dir', type=str, default=opt.root_dir, help='Batch size for training')
    parser.add_argument('--mode', type=str, default=opt.mode, help='Rgb or rgbd')
    parser.add_argument('--dataset', type=str, default=opt.dataset, help='aloha or glove')
    parser.add_argument('--train_file', type=str, default=opt.train_file, help='aloha or glove')
    parser.add_argument('--seed', type=int, default=opt.seed, help='aloha or glove')

    
    args = parser.parse_args()
    return args

def update_config_with_args(args):
    opt.root_dir = args.root_dir
    opt.mode = args.mode
    opt.train_file = args.train_file
    opt.dataset = args.dataset
    opt.seed = args.seed
    opt.joint_num = JOINT[opt.dataset]

    # 更新其他配置项...

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.data_rt = self.config.root_dir + "/" + self.config.dataset
        if self.config.model_save == '':
            self.model_save = self.config.net + '_ips' + str(
                self.config.input_size)

            self.model_dir = './checkpoint/' + self.config.root_dir + '/' + f'{self.config.seed}' + '/' + self.config.train_file + '/' + self.model_save

            print(f"Model will be saved to: {self.model_dir}")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.model_dir + '/img')
            os.makedirs(self.model_dir + '/debug')
            os.makedirs(self.model_dir + '/files')

        # save core file
        shutil.copyfile('train.py', self.model_dir + '/files/train.py')
        shutil.copyfile('./config.py', self.model_dir + '/files/config.py')
        shutil.copyfile('model/model.py', self.model_dir + '/files/model.py')
        #shutil.copyfile('./model/fusion_layer.py', self.model_dir + '/files/fusion_layer.py')
        shutil.copyfile('./dataloader/loader.py', self.model_dir + '/files/loader.py')

        # save config
        with open(self.model_dir + '/config.json', 'w') as f:
            for k, v in self.config.__class__.__dict__.items():
                if not k.startswith('_'):
                    print(str(k) + ":" + str(v))
                    f.writelines(str(k) + ":" + str(v) + '\n')

        cudnn.benchmark = False
        self.dataset = 'nyu_all' if 'nyu' in self.config.dataset else 'hands'
        self.joint_num = 23 if 'nyu' in self.config.dataset else self.config.joint_num

        self.net = RGBDmodel(self.config.net, self.config.pretrain, self.joint_num, self.dataset,
                            './MANO/', kernel_size=self.config.feature_para[0],mode=self.config.mode)

        self.net = DataParallel(self.net).cuda()
        self.GFM_ = GFM()

        optimList = [{"params": self.net.parameters(), "initial_lr": self.config.lr}]
        # init optimizer
        if self.config.opt == 'sgd':
            self.optimizer = SGD(optimList, lr=self.config.lr, momentum=0.9, weight_decay=1e-4)
        elif self.config.opt == 'adam':
            self.optimizer = Adam(optimList, lr=self.config.lr)
        elif self.config.opt == 'adamw':
            self.optimizer = AdamW(optimList, lr=self.config.lr, weight_decay=0.01)

        self.L1Loss = SmoothL1Loss().cuda()
        self.BECLoss = torch.nn.BCEWithLogitsLoss().cuda()

        self.L2Loss = torch.nn.MSELoss().cuda()
        self.start_epoch = 0

        # load model
        if self.config.load_model != '':
            print('loading model from %s' % self.config.load_model)
            checkpoint = torch.load(self.config.load_model, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # fine-tune model
        if self.config.finetune_dir != '':
            print('loading model from %s' % self.config.finetune_dir)
            checkpoint = torch.load(self.config.finetune_dir, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # init scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=0.1, last_epoch=self.start_epoch)

        if self.config.dataset == 'dexycb':
            if self.config.phase == 'train':
                self.trainData = loader.DexYCBDataset(self.config.dexycb_setup, 'train', self.config.root_dir,
                                                      aug_para=self.config.augment_para,
                                                      img_size=self.config.input_size)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=8)
            self.testData = loader.DexYCBDataset(self.config.dexycb_setup, 'test', self.config.root_dir,
                                                 img_size=self.config.input_size)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=8)

        elif self.config.dataset == 'ho3d':
            if 'train' in self.config.phase:
                self.trainData = loader.HO3D('train_all', self.config.root_dir,
                                             dataset_version=config.ho3d_version,
                                             aug_para=self.config.augment_para,
                                             img_size=self.config.input_size,
                                             cube_size=self.config.cube_size, color_factor=self.config.color_factor,
                                             center_type='joint_mean')
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=4)
            self.testData = loader.HO3D('test', self.config.root_dir, dataset_version=config.ho3d_version,
                                        img_size=self.config.input_size, cube_size=self.config.cube_size,
                                        center_type='joint_mean', aug_para=[0, 0, 0])
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=4)

            self.evalData = loader.HO3D('eval', self.config.root_dir, dataset_version=config.ho3d_version,
                                        img_size=self.config.input_size, cube_size=self.config.cube_size,

                                        aug_para=[0, 0, 0])
            self.evalLoader = DataLoader(self.evalData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=4)

        elif self.config.dataset == 'nyu':
            if 'train' in self.config.phase:
                self.trainData = loader.nyu_loader(self.data_rt, 'train', aug_para=self.config.augment_para,
                                                   img_size=self.config.input_size,
                                                   cube_size=self.config.cube_size,
                                                   center_type=self.config.center_type,
                                                   color_factor=self.config.color_factor)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=4)
            self.testData = loader.nyu_loader(self.data_rt, 'test', img_size=self.config.input_size,
                                              cube_size=self.config.cube_size,
                                              center_type=self.config.center_type, aug_para=[0, 0, 0])
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=4)
        elif self.config.dataset == 'STB':
            if self.config.phase == 'train':
                self.trainData = STB(self.config.dexycb_setup, 'train', self.config.root_dir,
                                     aug_para=self.config.augment_para,
                                     img_size=self.config.input_size)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=8)
            self.testData = STB(self.config.dexycb_setup, 'test', self.config.root_dir,
                                img_size=self.config.input_size)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=8)
        elif self.config.dataset == 'aloha':
            if self.config.phase == 'train':
                # print(self.config.augment_para)
                self.trainData = loader.AlohaDataset(self.config.root_dir, 'train', self.config.augment_para,
                                     
                                     img_size=self.config.input_size,mode=self.config.mode,crop=self.config.crop,train_file = self.config.train_file)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                            num_workers=8)
            self.testData = loader.AlohaDataset(self.config.root_dir, 'test', self.config.augment_para,
                                img_size=self.config.input_size,mode=self.config.mode,crop=self.config.crop)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                            num_workers=8)
            self.evalData = loader.AlohaDataset(self.config.root_dir, 'evaluation', self.config.augment_para,
                                            img_size=self.config.input_size,mode=self.config.mode,crop=self.config.crop)
            self.evalLoader = DataLoader(self.evalData, batch_size=1, shuffle=False,
                                         num_workers=8)
        else:
            raise NotImplementedError()

        self.test_error = 10000
        self.min_error = 100

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')
        self.min_error = 100
        self.writer = SummaryWriter('runs/' +  self.config.root_dir + '/' + self.config.mode + '/')

    def train(self):
        self.phase = 'train'
        for epoch in range(self.start_epoch, self.config.max_epoch):
            self.net.train()
            data_iter = tqdm(enumerate(self.trainLoader), total=len(self.trainLoader), desc=f'Epoch {epoch+1}/{self.config.max_epoch}')
            for ii, data in data_iter:

                img_rgb = data['image_rgb'].cuda()
                xyz_gt = data['joint_xyz'].cuda()

                if 'D' in self.config.mode:
                    dimg = data['image_depth'].cuda()
                    pcl = data['pcl'].cuda()

                else:
                    dimg = None  # No depth image in RGB mode
                    pcl = None

                self.optimizer.zero_grad()                    
                iter_num = ii + (self.trainData.__len__() // self.config.batch_size) * epoch

                results, _, _ = self.net(img_rgb, dimg, pcl)
                loss = 0
                for index, stage_type in enumerate(self.config.stage_type):

                    if stage_type == 1:  # pixel-wise backbone
                        pixel_pd = results[index]  # B x J x 3
                        loss_joint = self.L1Loss(xyz_gt, pixel_pd) 
                        # print(f'xyz_gt: {xyz_gt[0,:,0]}')
                        # print(f'pixel_pd: {pixel_pd[0,:,0]}')
                        loss += (loss_joint)

                        self.writer.add_scalar('train_loss', loss, global_step=iter_num)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
                self.optimizer.step()

            test_error = self.test(epoch)

            if test_error <= self.min_error:
                self.min_error = test_error
                save = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(
                    save,
                    self.model_dir + "/best.pth"
                )
            save = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }

            torch.save(
                save,
                self.model_dir + "/latest.pth"
            )

            if self.config.scheduler == 'auto':
                self.scheduler.step(test_error)
            elif self.config.scheduler == 'step':
                self.scheduler.step(epoch)
            elif self.config.scheduler == 'multi_step':
                self.scheduler.step()
            else:
                pass

    @torch.no_grad()
    def test(self, epoch=-1):
        self.phase = 'test'
        self.result_file_list = []
        for index in range(len(self.config.stage_type)):
            self.result_file_list.append(open(self.model_dir + '/test_%d.txt' % (index), 'w'))
        self.id_file = open(self.model_dir + '/id.txt', 'w')
        self.mano_file = open(self.model_dir + '/eval_mano.txt', 'w')
        self.net.eval()
        batch_num = 0
        error_list_all_batch = []
        error_list = [0] * len(self.config.stage_type)
        PA_error_list = [0] * len(self.config.stage_type)
        pck_list_01 = [0] * len(self.config.stage_type)
        pck_list_0075 = [0] * len(self.config.stage_type)
        pck_list_005 = [0] * len(self.config.stage_type)
        pck_list_0025 = [0] * len(self.config.stage_type)
        for ii, data in tqdm(enumerate(self.testLoader)):

            img_rgb = data['image_rgb'].cuda()
            xyz_gt = data['joint_xyz'].cuda()

            if 'D' in self.config.mode:
                dimg = data['image_depth'].cuda()
                pcl = data['pcl'].cuda()

            else:
                dimg = None  # No depth image in RGB mode
                pcl = None

            results, _, _ = self.net(img_rgb, dimg, pcl)

            batch_num += 1
            joint_error_list = []
            # PA_joint_error_list = []
            # pck_list_01 = []
            # pck_list_0075 = []
            # pck_list_005 = []
            # pck_list_0025 = []

            for index, stage_type in enumerate(self.config.stage_type):

                pixel_pd = results[index]
                joint_xyz = pixel_pd
                print(f'xyz_gt: {xyz_gt[0,:,0]}')
                print(f'joint_xyz: {pixel_pd[0,:,0]}')
                joint_errors = self.xyz2error(joint_xyz, xyz_gt, self.result_file_list[index])
                batch_errors = np.mean(joint_errors, axis=-1)

                pck_errors_01 = self.pck(joint_xyz, xyz_gt, threshold=0.01)
                pck_errors_0075 = self.pck(joint_xyz, xyz_gt, threshold=0.0075)
                pck_errors_005 = self.pck(joint_xyz, xyz_gt, threshold=0.005)
                pck_errors_0025 = self.pck(joint_xyz, xyz_gt, threshold=0.0025)
                batch_errors_pck01 = np.mean(pck_errors_01, axis=-1)
                batch_errors_pck0075 = np.mean(pck_errors_0075, axis=-1)
                batch_errors_pck005 = np.mean(pck_errors_005, axis=-1)
                batch_errors_pck0025 = np.mean(pck_errors_0025, axis=-1)

                joint_errors_aligned = 0
                for b in range(joint_xyz.shape[0]):

                    j, j_gt = joint_xyz[b].cpu().numpy(), xyz_gt[b].cpu().numpy()
                    if self.dataset != 'STB':
                        joint_xyz_aligned = self.GFM_.rigid_align(j, j_gt)
                    else:
                        joint_xyz_aligned = joint_xyz - (joint_xyz[0] - xyz_gt[0])

                    joint_errors_aligned += self.xyz2error(torch.from_numpy(joint_xyz_aligned).cuda().unsqueeze(0),
                                                            xyz_gt[b].unsqueeze(0), self.result_file_list[index])
                batch_errors_aligned = joint_errors_aligned / joint_xyz.shape[0]


                joint_error_list.append(joint_errors)
                error = np.mean(batch_errors)
                error_list[index] += error

                PA_error = np.mean(batch_errors_aligned)
                PA_error_list[index] += PA_error

                pck_list_01[index] += np.mean(batch_errors_pck01)
                pck_list_0075[index] += np.mean(batch_errors_pck0075)
                pck_list_005[index] += np.mean(batch_errors_pck005)
                pck_list_0025[index] += np.mean(batch_errors_pck0025)

            error_list_all_batch.append(joint_error_list)
        eval_auc(error_list_all_batch, self.testLoader.__len__())
        error_info = '%d epochs:  ' % epoch
        for index in range(len(error_list)):
            print("[mean_Error %.6f]" % (error_list[index] / batch_num))
            error_info += ' error' + str(index) + ": %.6f" % (error_list[index] / batch_num) + ' '

            print("[PA_mean_Error %.6f]" % (PA_error_list[index] / batch_num))
            error_info += ' PA_error' + str(index) + ": %.6f" % (PA_error_list[index] / batch_num) + ' '

            # 输出 PCK 到 log
            print("[PCK_0.1] %.6f" % (pck_list_01[index] / batch_num))
            error_info += ' PCK_0.1_stage' + str(index) + ": %.6f" % (pck_list_01[index] / batch_num) + ' '

            print("[PCK_0.075] %.6f" % (pck_list_0075[index] / batch_num))
            error_info += ' PCK_0.075_stage' + str(index) + ": %.6f" % (pck_list_0075[index] / batch_num) + ' '

            print("[PCK_0.05] %.6f" % (pck_list_005[index] / batch_num))
            error_info += ' PCK_0.05_stage' + str(index) + ": %.6f" % (pck_list_005[index] / batch_num) + ' '

            print("[PCK_0.025] %.6f" % (pck_list_0025[index] / batch_num))
            error_info += ' PCK_0.025_stage' + str(index) + ": %.6f" % (pck_list_0025[index] / batch_num) + ' '

        logging.info(error_info)

        self.writer.add_scalar('mean_Error', error_list[index] / batch_num, global_step=epoch)
        self.writer.add_scalar('PA_mean_Error', PA_error_list[index] / batch_num, global_step=epoch)
        self.writer.add_scalar('PCK_0.1', np.mean(pck_list_01) / batch_num, global_step=epoch)
        self.writer.add_scalar('PCK_0.075', np.mean(pck_list_0075) / batch_num, global_step=epoch)
        self.writer.add_scalar('PCK_0.05', np.mean(pck_list_005) / batch_num, global_step=epoch)
        self.writer.add_scalar('PCK_0.025', np.mean(pck_list_0025) / batch_num, global_step=epoch)


        return error_list[-1] / batch_num

    @torch.no_grad()
    def evalution(self, epoch=-1):
        self.phase = 'evaluation'
        self.net.eval()
        joint_list = []
        mesh_list= []
        total_time = 0
        total_frames = 0
        print(f"Processing {len(self.evalLoader)} files.")

        for ii, data in tqdm(enumerate(self.evalLoader)):
            start_time = time.time()
            img_rgb = data['image_rgb'].cuda()
            xyz_gt = data['joint_xyz'].cuda()

            if 'D' in self.config.mode:
                dimg = data['image_depth'].cuda()
                pcl = data['pcl'].cuda()

            else:
                dimg = None  # No depth image in RGB mode
                pcl = None

            results, _, _ = self.net(img_rgb, dimg, pcl)

            batch_size = img_rgb.size(0)
 
            joint_xyz_list = []
            for index, stage_type in enumerate(self.config.stage_type):

                pixel_pd = results[index]
                joint_xyz = pixel_pd
                joint_xyz_list.append(joint_xyz.cpu().detach().numpy()) 


            # print(f'joint_xyz {joint_xyz[0][0]}')
            joint_xyz_array = np.concatenate(joint_xyz_list, axis=0)
            # print(f'joint_xyz_array{joint_xyz_array}')
            # print(f'xyz_gt {xyz_gt[0][0]}')
            joint_list = joint_list + np.split(joint_xyz_array, batch_size, axis=0)
            mesh_list = mesh_list + np.split(xyz_gt, batch_size, axis=0)

            frame_time = time.time() - start_time
            total_time += frame_time
            total_frames += 1

            # 输出每一帧的处理时间
            # print(f"Processing frame {ii + 1}/{len(self.evalLoader)} took {frame_time:.4f} seconds.")

        self.dump(self.model_dir + '/pred.json', joint_list, mesh_list)
        return 0

    @torch.no_grad()
    def xyz2error(self, output, joint, write_file=None):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        batch_size, joint_num, _ = output.shape

        # 计算预测的三维关节坐标与真实的三维关节坐标之间的平方差
        errors = (output - joint) ** 2

        # 对平方差求和并开方，得到每个关节的欧几里得距离（误差）
        errors = np.sqrt(np.sum(errors, axis=2))

        return errors
    
    @torch.no_grad()
    def pck(self, output, joint, threshold=0.05, write_file=None):
        # output = output.detach().cpu().numpy()
        # joint = joint.detach().cpu().numpy()
        # batch_size, joint_num, _ = output.shape

        # 计算每个关节的欧几里得距离
        errors = self.xyz2error(output, joint)

        # 判断每个关节的预测是否在阈值范围内
        correct = (errors < threshold).astype(np.float32)

        # 计算PCK：每个样本的正确关键点比例
        pck = np.mean(correct, axis=1)

        # 返回平均PCK值
        return pck


    @torch.no_grad()
    def z2error(self, output, joint, center, cube_size, write_file=None):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center

        # errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
        # errors = np.sqrt(np.sum(errors, axis=2))
        # print(joint_xyz.shape)
        depth_errors = abs((joint_xyz - joint_world_select)[:, :, 2])
        return depth_errors

    @torch.no_grad()
    def xy2error(self, output, joint, center, cube_size, write_file=None):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center

        xy_errors = (joint_xyz - joint_world_select)[:, :, :2] * (joint_xyz - joint_world_select)[:, :, :2]
        xy_errors = np.sqrt(np.sum(xy_errors, axis=2))
        # print(joint_xyz.shape)
        # depth_errors = abs((joint_xyz - joint_world_select)[:, :, 2])
        return xy_errors

    import json

    def dump(self, pred_out_path, xyz_pred_list, label_list):
        """ Save predictions into a json file, clearly distinguishing predictions and ground truth. """
        
        # Make sure the input lists are in the correct format (convert tensors to lists)
        xyz_pred_list = [x[0].tolist() for x in xyz_pred_list]
        label_list = [x[0].tolist() for x in label_list]
        
        # Prepare the formatted data: we will store predictions and ground truth separately
        formatted_xyz_preds = []
        formatted_label_data = []
        
        # Process the predictions and labels to store them with 'pred' and 'gt' type
        for idx, (xyz_pred, label) in enumerate(zip(xyz_pred_list, label_list)):
            # Add the joint predictions (xyz) with 'pred' type
            formatted_xyz_preds.append({
                "id": idx,  # Each prediction has an id
                "type": "pred",  # Mark as a prediction
                "xyz": xyz_pred  # Add the predicted xyz coordinates
            })
            
            # Add the ground truth labels (xyz) with 'gt' type
            formatted_label_data.append({
                "id": idx,  # Each label also has the same id for alignment
                "type": "gt",  # Mark as a ground truth
                "xyz": label  # Add the ground truth xyz coordinates
            })
        
        # Save the formatted data to a JSON file
        with open(pred_out_path, 'w') as fo:
            json.dump({
                "xyz_predictions": formatted_xyz_preds,  # Predictions for xyz coordinates
                "ground_truth": formatted_label_data  # Ground truth for xyz coordinates
            }, fo, indent=4)  # Pretty print with indentation for readability

        # Print the number of predictions and labels saved to the file
        print(f'Dumped {len(formatted_xyz_preds)} joint predictions and {len(formatted_label_data)} ground truth joints to {pred_out_path}')


if __name__ == '__main__':
    set_seed(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    args = parse_args()
    
    # 更新配置
    update_config_with_args(args)

    Trainer = Trainer(opt)
    # if 'train' in Trainer.config.phase:
    #     Trainer.train()
    #     Trainer.writer.close()
    # elif Trainer.config.phase == 'test':
    #     Trainer.test()
    #     # Trainer.result_file.close()
    # elif Trainer.config.phase == 'eval':
        # Trainer.evalution()
    Trainer.train()
    Trainer.writer.close()
    # Trainer.evalution()