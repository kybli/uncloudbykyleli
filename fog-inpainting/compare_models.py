import argparse
from re import A
import numpy as np
import os

from yaml import parse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--use-edge-metrics', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    models = ['AOD-Net', 'DCP', 'DehazeFormer-S', 'FFA-Net', 'GridDehazeNet', 'UNCLOUD']

    args = parse_args()

    mse = []
    psnr = []
    ssim = []

    metric_dict = {"MSE:": mse, "PSNR:": psnr, "SSIM:": ssim}


    edge_mse = []
    edge_psnr = []
    edge_ssim = []

    edge_metric_dict = {"MSE:": edge_mse, "PSNR:": edge_psnr, "SSIM:": edge_ssim}

    # PARSE METFICS FILES FOR EACH MODEL LISTED ABOVE IN 'models' LIST
    for i in range(len(models)):
        model = models[i]

        metric_dict['MSE:'].append([])
        metric_dict['PSNR:'].append([])
        metric_dict['SSIM:'].append([])

        edge_metric_dict['MSE:'].append([])
        edge_metric_dict['PSNR:'].append([])
        edge_metric_dict['SSIM:'].append([])

        metric_file_path = os.path.join(args.dir, model, "metrics.txt")
        edge_metric_file_path = os.path.join(args.dir, model, "edge_metrics.txt")
        
        
        # PARSE NON EDGE METRIC FILE
        metric_file = open(metric_file_path, 'r')
        lines = metric_file.readlines()

        read_cur_line = False

        for line in lines:
            if line == "FOGGY IMAGE STATS:\n":
                read_cur_line = False
            elif line == "DEFOGGED IMAGE STATS:\n":
                read_cur_line = True
            elif read_cur_line:
                parsed_line = line.split(" ")
                
                if len(parsed_line) == 1:
                    continue

                metric = parsed_line[0].split("\n")[0]
                value = parsed_line[1].split("\n")[0]

                metric_dict[metric][i].append(value)
            
        

        # PARSE EDGE METRIC FILE
        metric_file = open(edge_metric_file_path, 'r')
        lines = metric_file.readlines()

        read_cur_line = False

        for line in lines:
            if line == "FOGGY IMAGE STATS:\n":
                read_cur_line = False
            elif line == "DEFOGGED IMAGE STATS:\n":
                read_cur_line = True
            elif read_cur_line:
                parsed_line = line.split(" ")
                
                if len(parsed_line) == 1:
                    continue

                metric = parsed_line[0].split("\n")[0]
                value = parsed_line[1].split("\n")[0]

                edge_metric_dict[metric][i].append(value)
            
        
    # PREPARATION FOR COMPARISON
    for key in metric_dict.keys():
        metric_dict[key] = np.asarray(metric_dict[key], float)
    
    for key in edge_metric_dict.keys():
        edge_metric_dict[key] = np.asarray(edge_metric_dict[key], float)

    avg_dict = {"avg_mse": None, "avg_psnr": None, "avg_ssim": None, "avg_edge_mse": None, "avg_edge_psnr": None, "avg_edge_ssim": None}

    avg_dict['avg_mse'] = np.average(metric_dict['MSE:'], axis=1)
    avg_dict['avg_psnr'] = np.average(metric_dict['PSNR:'], axis=1)
    avg_dict['avg_ssim'] = np.average(metric_dict['SSIM:'], axis=1)

    avg_dict['avg_edge_mse'] = np.average(edge_metric_dict['MSE:'], axis=1)
    avg_dict['avg_edge_psnr'] = np.average(edge_metric_dict['PSNR:'], axis=1)
    avg_dict['avg_edge_ssim'] = np.average(edge_metric_dict['SSIM:'], axis=1)


    # COMPARISON BETWEEN MODELS
    for key in avg_dict.keys():
        if not args.use_edge_metrics and key in ["avg_edge_mse", "avg_edge_psnr", "avg_edge_ssim"]:
            continue

        if key == "avg_mse" or key == "avg_edge_mse":
            best_idx = np.argmin(avg_dict[key])
        else:
            best_idx = np.argmax(avg_dict[key])
        
        print("{} ------------------------------".format(key))
        for i in range(len(avg_dict[key])):
            val = format(np.around(avg_dict[key][i], decimals=5), '.5f')

            if i == best_idx:
                print("{}  [WINNER]             {}".format(val, models[i]))
            else:
                print("{}                       {}".format(val, models[i]))
        
        print("\n\n")
    

    # USE FOLLOWING INSTEAD OF ABOVE 'FOR LOOP' TO GENERATE LaTeX TABLE CONTAINING COMPARISON
    '''
    key_dict = {"avg_mse": "MSE", "avg_psnr": "PSNR", "avg_ssim": "SSIM"}

    for key in avg_dict.keys():
        if key in ["avg_edge_mse", "avg_edge_psnr", "avg_edge_ssim"]:
            continue

        if key == "avg_mse" or key == "avg_edge_mse":
            best_idx = np.argmin(avg_dict[key])
        else:
            best_idx = np.argmax(avg_dict[key])
        
        print("{} ------------------------------".format(key))
        for i in range(len(avg_dict[key])):
            print(models[i], end=', ')

        print('\n')

        print("& {} ".format(key_dict[key]), end='')
        for i in range(len(avg_dict[key])):
            print("& ", end='')

            val = format(np.around(avg_dict[key][i], decimals=2), '.2f')
            if i == best_idx:
                print("\\textbf{{{}}} ".format(val, models[i]), end='')
            else:
                print("{} ".format(val), end='')
            
        print("\\\\")
        
        print("\n\n")
        '''

