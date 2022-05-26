import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize

import numpy as np

import metrics

import skimage.io as io
import os
from skimage.util import img_as_ubyte

def evaluate(model, dataset, device, filename, save_grid=True, limit_img_count_to_eight=True, get_metrics=False, use_edge_metrics=True, save_dir= None, save_img=False, save_edge_img=False, **kwargs):
    #if save_grid and not limit_img_count_to_eight:
    #    raise Exception("to save grid, need to limit img count to eight")
    
    img_count_to_eval = len(dataset)
    if limit_img_count_to_eight:
        img_count_to_eval = 8
    
    image, mask, gt = zip(*[dataset[i] for i in range(img_count_to_eval)])
    image = torch.stack(image)
    transformed_input_image = image
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    if save_grid:
        grid = make_grid(
            torch.cat((unnormalize(image), mask, unnormalize(output),
                    unnormalize(output_comp), unnormalize(gt)), dim=0))
        save_image(grid, filename)


    if get_metrics:
        image = unnormalize(image)
        transformed_input_image = unnormalize(transformed_input_image)
        output = unnormalize(output)
        output_comp = unnormalize(output_comp)
        gt = unnormalize(gt)

        mse_defogged_arr = np.array([])
        ssim_defogged_arr = np.array([])
        psnr_defogged_arr = np.array([])

        if kwargs.get('metric_file'):
            metric_file_path = kwargs.get('metric_file')
            metric_file = open(metric_file_path, 'a')
            metric_file.write("-----------------------------------------------------------------------------------------------------------\n")
            
            if 'pth_num' in kwargs:
                metric_file.write("{}pth\n\n".format(kwargs['pth_num']))

        for i in range(img_count_to_eval):
            transformed_input_img = transformed_input_image[i].data.cpu().numpy().squeeze().transpose(1, 2, 0)
            transformed_input_img = np.asarray(transformed_input_img.clip(0, 1)*255, dtype=np.uint8)

            gt_img = gt[i].data.cpu().numpy().squeeze().transpose(1, 2, 0)
            gt_img = np.asarray(gt_img.clip(0, 1)*255, dtype=np.uint8)

            output_img = output[i].data.cpu().numpy().squeeze().transpose(1, 2, 0)
            output_img = np.asarray(output_img.clip(0, 1)*255, dtype=np.uint8)

            if use_edge_metrics == False:
                mse_fog, ssim_fog, psnr_fog = metrics.edge_detection_metrics(transformed_input_img, gt_img, return_edges=False, use_edge=False)
                mse_defogged, ssim_defogged, psnr_defogged = metrics.edge_detection_metrics(output_img, gt_img, return_edges=False, use_edge=False)

            else:
                mse_fog, ssim_fog, psnr_fog, edge_fog, edge_gt = metrics.edge_detection_metrics(transformed_input_img, gt_img, return_edges=True, use_edge=True)
                mse_defogged, ssim_defogged, psnr_defogged, edge_defogged, _ = metrics.edge_detection_metrics(output_img, gt_img, return_edges=True, use_edge=True)

                if save_img:
                    if save_dir is None:
                        raise Exception("need save_dir to be specified to save imgs")
                    
                    gt_edge_dir = os.path.join(save_dir, "img/edge/gt")
                    input_edge_dir = os.path.join(save_dir, "img/edge/input")
                    output_edge_dir = os.path.join(save_dir, "img/edge/output")

                    if not os.path.isdir(gt_edge_dir):
                        os.makedirs(gt_edge_dir)
                    if not os.path.isdir(input_edge_dir):
                        os.makedirs(input_edge_dir)
                    if not os.path.isdir(output_edge_dir):
                        os.makedirs(output_edge_dir)
                    
                    io.imsave(os.path.join(gt_edge_dir, '{}_gt_edge.jpg'.format(i)), img_as_ubyte(edge_gt))
                    io.imsave(os.path.join(input_edge_dir, '{}_fog_edge.jpg'.format(i)), (edge_fog * 255).astype(np.uint8))
                    io.imsave(os.path.join(output_edge_dir, '{}_defog_edge.jpg'.format(i)), (edge_defogged * 255).astype(np.uint8))


            mse_defogged_arr = np.append(mse_defogged_arr, mse_defogged)
            ssim_defogged_arr = np.append(ssim_defogged_arr, ssim_defogged)
            psnr_defogged_arr = np.append(psnr_defogged_arr, psnr_defogged)

            if kwargs.get('metric_file'):
                metric_file.write("[{}]\n\n".format(i))
                metric_file.write("FOGGY IMAGE STATS:\n")
                metric_file.write("MSE: {}\n".format(mse_fog))
                metric_file.write("PSNR: {}\n".format(psnr_fog))
                metric_file.write("SSIM: {}\n\n".format(ssim_fog))
                
                metric_file.write("DEFOGGED IMAGE STATS:\n")
                metric_file.write("MSE: {}\n".format(mse_defogged))
                metric_file.write("PSNR: {}\n".format(psnr_defogged))
                metric_file.write("SSIM: {}\n\n".format(ssim_defogged))

                metric_file.write("\n\n\n")

            if save_edge_img:
                if save_dir is None:
                    raise Exception("need save_dir to be specified to save imgs")
                
                edge_gt_dir = os.path.join(save_dir, "img/edge/gt")
                edge_input_dir = os.path.join(save_dir, "img/edge/input")
                edge_output_dir = os.path.join(save_dir, "img/edge/output")

                if not os.path.isdir(edge_gt_dir):
                    os.makedirs(edge_gt_dir)
                if not os.path.isdir(edge_input_dir):
                    os.makedirs(edge_input_dir)
                if not os.path.isdir(edge_output_dir):
                    os.makedirs(edge_output_dir)

                io.imsave(os.path.join(edge_gt_dir, "{}_gt_edge.jpg".format(i)), img_as_ubyte(edge_gt))
                io.imsave(os.path.join(edge_input_dir, "{}_fog_edge.jpg".format(i)), (edge_fog * 255).astype(np.uint8))
                io.imsave(os.path.join(edge_output_dir, "{}_defog_edge.jpg".format(i)), (edge_defogged * 255).astype(np.uint8))


    if save_img:
        if save_dir is None:
            raise Exception("need save_dir to be specified to save imgs")
        for i in range(img_count_to_eval):
            transformed_input_img = transformed_input_image[i]
            gt_img = gt[i]
            output_img = output[i]

            gt_dir = os.path.join(save_dir, "img/gt")
            input_dir = os.path.join(save_dir, "img/input")
            output_dir = os.path.join(save_dir, "img/output")

            if not os.path.isdir(gt_dir):
                os.makedirs(gt_dir)
            if not os.path.isdir(input_dir):
                os.makedirs(input_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            save_image(gt_img, os.path.join(gt_dir, "{}_gt.jpg".format(i)))
            save_image(transformed_input_img, os.path.join(input_dir, "{}_input.jpg".format(i)))
            save_image(output_img, os.path.join(output_dir, "{}_output.jpg".format(i)))

    if get_metrics:
        return mse_defogged_arr, ssim_defogged_arr, psnr_defogged_arr
