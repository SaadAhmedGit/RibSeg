"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import sys
import importlib
from tqdm import tqdm
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=250000, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir


    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    data_list = tqdm([x for x in os.listdir('./data/pn/data_pn/')])
    

    with torch.no_grad():
        time_cost = 0
        num=0
        ave_dice=torch.tensor(0).float().cuda()
        for ct in data_list:
            num+=1
            data = np.load('./data/pn/data_pn/'+ct).astype(np.float32)
            seg = np.load('./data/pn/label_pn/'+ct).astype(np.int32)
            seg[seg!=0]=1

            points = data[:, 0:3]
            choice = np.random.choice(data.shape[0], 30000, replace=False)
            # resample
            points = points[choice, :]
            seg = seg[choice]
            np.save('./inference_res/point/'+ct,points.astype('int32'))

            points[:, 0:3] = pc_normalize(points[:, 0:3])

            label = np.array([0])

            points = np.expand_dims(points, 0)
            label = np.expand_dims(label, 0)
            
            points, label,seg = torch.from_numpy(points).float().cuda(), torch.from_numpy(label).long().cuda(),torch.from_numpy(seg).long().cuda()
            points = points.transpose(2, 1)

            t1=time.perf_counter()

            classifier = classifier.eval()
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))

            time_cost += time.perf_counter() - t1
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            pred_choice = seg_pred.data.max(1)[1]

            ## dice
            intersection = pred_choice.mul(seg)
            
            i_s = torch.sum(intersection)
            p_s = torch.sum(pred_choice)
            t_s = torch.sum(seg)
            dice = 1- (2*(i_s)+1)/(p_s+t_s+1)
            ave_dice=torch.add(ave_dice,dice)
            
            ####################################

            pred_choice=pred_choice.cpu().numpy()
            np.save('./inference_res/label/'+ct[:-4],pred_choice.astype('int8'))
        time_cost/=num
        ave_dice/=num
        print('average time:',time_cost)
        print('average dice:',ave_dice)

if __name__ == '__main__':
    args = parse_args()
    main(args)
