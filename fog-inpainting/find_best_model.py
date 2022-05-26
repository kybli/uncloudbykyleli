import argparse
import torch
from torchvision import transforms

import opt
from data_loader import data_loader
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt

import numpy as np
import os
import logging

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--snapshot-dir', type=str, default='')
    parser.add_argument('--pth-interval', type=int, default=1000)
    parser.add_argument('--max-pth', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--metric-file', type=str, default=None)
    
    return parser.parse_args()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()
    
    device = torch.device('cuda')
    size = (args.image_size, args.image_size)

    img_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(),
        transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])
    
    

    datasets = ['cityscapes', 'nitre', 'frida', 'frida2', 'small-aerial', 'dense-haze', 'o-hazy-nitre', 'i-hazy-nitre']
    model = PConvUNet().to(device)

    evaluation_results = {'overall': {'best_model': {'by_avg_mse': {'model': 0, 'avg_mse': 0}, 
                                                      'by_avg_ssim': {'model': 0, 'avg_ssim': 0},
                                                      'by_avg_psnr': {'model': 0, 'avg_psnr': 0}},
                                       'avg_mse_arr_by_pth': None,
                                       'avg_ssim_arr_by_pth': None,
                                       'avg_psnr_arr_by_pth': None}}

    max_pth = args.max_pth
    pth_interval = args.pth_interval

    for dataset in datasets:
        logging.info("--------  LOADING {} Dataset  --------".format(dataset))
        dataset_val = data_loader(args.root, img_transform, mask_transform, 'val', include=[dataset], abs_dataset_cap=50)

        evaluation_results[dataset] = {'best_model': {'by_avg_mse': {'model': None, 'avg_mse': None}, 
                                                      'by_avg_ssim': {'model': None, 'avg_ssim': None},
                                                      'by_avg_psnr': {'model': None, 'avg_psnr': None}},
                                       'avg_mse_arr_by_pth': np.array([]),
                                       'avg_ssim_arr_by_pth': np.array([]),
                                       'avg_psnr_arr_by_pth': np.array([])}

        for pth in range(pth_interval, max_pth + 1, pth_interval):
            pth_path = os.path.join(args.snapshot_dir, "{}.pth".format(str(pth)))

            if not os.path.isfile(pth_path):
                if pth == max_pth:
                    raise Exception("[max_pth].pth is not a valid file. Retry with valid max_pth parameter.")
                continue

            logging.info("path for {} = {}".format("{}.pth".format(str(pth)), pth_path))

            load_ckpt(pth_path, [('model', model)])
            model.eval()

            mse_defogged_arr, ssim_defogged_arr, psnr_defogged_arr = evaluate(model, dataset_val, device, '',
                                                                              save_grid=False, limit_img_count_to_eight=False, get_metrics=True,
                                                                              pth_num=pth, metric_file = args.metric_file)

            avg_mse = np.average(mse_defogged_arr)
            avg_ssim = np.average(ssim_defogged_arr)
            avg_psnr = np.average(psnr_defogged_arr)

            evaluation_results[dataset]['avg_mse_arr_by_pth'] = np.append(evaluation_results[dataset]['avg_mse_arr_by_pth'], avg_mse)
            evaluation_results[dataset]['avg_ssim_arr_by_pth'] = np.append(evaluation_results[dataset]['avg_ssim_arr_by_pth'], avg_ssim)
            evaluation_results[dataset]['avg_psnr_arr_by_pth'] = np.append(evaluation_results[dataset]['avg_psnr_arr_by_pth'], avg_psnr)

            if evaluation_results[dataset]['best_model']['by_avg_mse']['avg_mse'] == None or avg_mse < evaluation_results[dataset]['best_model']['by_avg_mse']['avg_mse']:
                evaluation_results[dataset]['best_model']['by_avg_mse']['avg_mse'] = avg_mse
                evaluation_results[dataset]['best_model']['by_avg_mse']['model'] = pth
            
            if evaluation_results[dataset]['best_model']['by_avg_ssim']['avg_ssim'] == None or avg_ssim > evaluation_results[dataset]['best_model']['by_avg_ssim']['avg_ssim']:
                evaluation_results[dataset]['best_model']['by_avg_ssim']['avg_ssim'] = avg_ssim
                evaluation_results[dataset]['best_model']['by_avg_ssim']['model'] = pth
            
            if evaluation_results[dataset]['best_model']['by_avg_psnr']['avg_psnr'] == None or avg_psnr > evaluation_results[dataset]['best_model']['by_avg_psnr']['avg_psnr']:
                evaluation_results[dataset]['best_model']['by_avg_psnr']['avg_psnr'] = avg_psnr
                evaluation_results[dataset]['best_model']['by_avg_psnr']['model'] = pth
        
        
        # UPDATE OVERALL
        if evaluation_results['overall']['avg_mse_arr_by_pth'] is None:
            evaluation_results['overall']['avg_mse_arr_by_pth'] = evaluation_results[dataset]['avg_mse_arr_by_pth']
            evaluation_results['overall']['avg_ssim_arr_by_pth'] = evaluation_results[dataset]['avg_ssim_arr_by_pth']
            evaluation_results['overall']['avg_psnr_arr_by_pth'] = evaluation_results[dataset]['avg_psnr_arr_by_pth']

        evaluation_results['overall']['avg_mse_arr_by_pth'] += evaluation_results[dataset]['avg_mse_arr_by_pth']
        evaluation_results['overall']['avg_ssim_arr_by_pth'] += evaluation_results[dataset]['avg_ssim_arr_by_pth']
        evaluation_results['overall']['avg_psnr_arr_by_pth'] += evaluation_results[dataset]['avg_psnr_arr_by_pth']

    # turn overall sums into avgs
    dataset_count = len(datasets)

    evaluation_results['overall']['avg_mse_arr_by_pth'] /= dataset_count
    evaluation_results['overall']['avg_ssim_arr_by_pth'] /= dataset_count
    evaluation_results['overall']['avg_psnr_arr_by_pth'] /= dataset_count

    evaluation_results['overall']['best_model']['by_avg_mse']['avg_mse'] = np.amin(evaluation_results['overall']['avg_mse_arr_by_pth'])
    evaluation_results['overall']['best_model']['by_avg_mse']['model'] = (np.argmin(evaluation_results['overall']['avg_mse_arr_by_pth'] + 1) * pth_interval)

    evaluation_results['overall']['best_model']['by_avg_ssim']['avg_ssim'] = np.amax(evaluation_results['overall']['avg_ssim_arr_by_pth'])
    evaluation_results['overall']['best_model']['by_avg_ssim']['model'] = (np.argmax(evaluation_results['overall']['avg_ssim_arr_by_pth'] + 1) * pth_interval)

    evaluation_results['overall']['best_model']['by_avg_psnr']['avg_psnr'] = np.amax(evaluation_results['overall']['avg_psnr_arr_by_pth'])
    evaluation_results['overall']['best_model']['by_avg_psnr']['model'] = (np.argmax(evaluation_results['overall']['avg_psnr_arr_by_pth'] + 1) * pth_interval)


    # GRAPHING
    logging.info("--------  BEGINNING GRAPHING  --------")

    x = [pth for pth in range(pth_interval, args.max_pth + 1, pth_interval)]

    fig, axs = plt.subplots(3)
    metrics = ["mse", "ssim", "psnr"]

    line_color = ['gray',   'lightcoral', 'sandybrown',  'navajowhite', 'khaki', 'palegreen', 'paleturquoise', 'thistle']
    point_color = ['black', 'maroon',     'saddlebrown', 'darkorange',  'gold',  'darkgreen', 'blue',          'indigo']

    for metric_num in range(3):
        logging.info("--------  graphing {}  --------".format(metrics[metric_num]))
        metric = metrics[metric_num]

        for dataset_num in range(dataset_count):
            dataset = datasets[dataset_num]
            y = evaluation_results[dataset]['avg_{}_arr_by_pth'.format(metric)]

            best_x = evaluation_results[dataset]['best_model']['by_avg_{}'.format(metric)]['model']
            best_y = evaluation_results[dataset]['best_model']['by_avg_{}'.format(metric)]['avg_{}'.format(metric)]

            axs[metric_num].plot(x, y, linestyle='solid', color=line_color[dataset_num], label=dataset)
            axs[metric_num].plot(best_x, best_y, marker='o', color=point_color[dataset_num])
        
        y = evaluation_results['overall']['avg_{}_arr_by_pth'.format(metric)]
        best_x = evaluation_results['overall']['best_model']['by_avg_{}'.format(metric)]['model']
        best_y = evaluation_results['overall']['best_model']['by_avg_{}'.format(metric)]['avg_{}'.format(metric)]
        
        axs[metric_num].plot(x, y, linestyle='solid', color='red', label='overall')
        axs[metric_num].plot(best_x, best_y, marker='o', color='fuchsia')

        axs[metric_num].set_title(metric)

        print("{}:".format(metric))
        print("pth: {}".format(best_x))
        print("val: {}\n\n".format(best_y))
    
    plt.subplots_adjust(hspace=0.75, right=0.75)
    axs[0].legend(bbox_to_anchor=(1, 0.5), loc='center left')
    fig.savefig('plot.png')
    

