from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='iPASSR_2xSR')
    return parser.parse_args()


def test(cfg):
    net = Net(cfg.scale_factor).to(cfg.device)
    model = torch.load('./log/' + cfg.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])
    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor))
    for idx in range(len(file_list)):
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
        LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')
        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
        LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
        scene_name = file_list[idx]
        print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():
            SR_left, SR_right = net(LR_left, LR_right, is_training=0)
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L.png')
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R.png')


if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Flickr1024', 'KITTI2012', 'KITTI2015', 'Middlebury']
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
