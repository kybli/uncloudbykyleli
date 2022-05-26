import os
import torch

from PIL import Image
from glob import glob
import logging

class data_loader(torch.utils.data.Dataset):
    def __init__(self, data_root, img_transform, mask_transform,
                 split='train', exclude=[], include=[], one_val_per_dataset=False, cap_cityscapes=True, abs_dataset_cap=None):
        super(data_loader, self).__init__()

        logging.getLogger().setLevel(logging.INFO)
        logging.info("########  COMMENCING DATA LOADER INIT  ########\n\n")

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        if split != 'train' and split != 'test' and split != 'val':
            raise Exception("split must be either 'train', 'test', or 'val'")
        
        self.fog_paths = []
        self.gt_paths = []
        self.mask_paths = []

        for dataset in os.listdir(data_root):
            if dataset in exclude or (include != [] and dataset not in include):
                logging.info("--------  SKIPPING {} Dataset  --------".format(dataset))
                continue

            logging.info("--------  Parsing {} Dataset  --------".format(dataset))
            dataset_path = os.path.join(data_root, dataset)
            
            foggy_dataset_name = "foggy-" + dataset
            gt_dataset_name = "gt-" + dataset
            mask_dataset_name = "mask-" + dataset

            foggy_dataset_path = os.path.join(dataset_path, foggy_dataset_name)
            gt_dataset_path = os.path.join(dataset_path, gt_dataset_name)
            mask_dataset_path = os.path.join(dataset_path, mask_dataset_name)
            
            # data must be stored as data_root/[dataset i.e. cityscapes]/[speciifc type of data for dataset i.e. fog, gt, or mask]/**/[train/test/val]/**/*.png
            # Each dataset contains a fog, gt, and mask folder. within each of these is a test/train/val that may be separated by several folders
            foggy_data_file_paths = glob('{:s}/**/{}/**/*.png'.format(foggy_dataset_path, split), recursive=True)

            if split == 'train':
                foggy_data_file_paths.extend(glob('{:s}/**/{}/**/*.png'.format(foggy_dataset_path, 'test'), recursive=True))

            num_of_files_added_from_cur_dataset = 0

            for foggy_data_file_path in foggy_data_file_paths:
                # dataset_path + "other data set name (depends on mask, gt, etc" + rel path minus name (foggy_data_file_rel_path_prefix) + "file name (depending on mask, gt, etc"
                foggy_data_file_path_prefix, foggy_data_file_name = os.path.split(foggy_data_file_path)
                data_file_rel_path_prefix = os.path.relpath(foggy_data_file_path_prefix, start=foggy_dataset_path)

                gt_data_file_name = foggy_data_file_name
                mask_data_file_name = foggy_data_file_name[:-4] + "_mask.png"

                if dataset == "cityscapes":
                    split_foggy_data_file_name = foggy_data_file_name.split("_")
                    gt_data_file_name = split_foggy_data_file_name[0] + "_" + split_foggy_data_file_name[1] + "_" + split_foggy_data_file_name[2] + "_" + split_foggy_data_file_name[3] + ".png"

                if dataset == "frida" or dataset == "frida2":
                    split_foggy_data_file_name = foggy_data_file_name.split("-")
                    gt_data_file_name = "LIma" + "-" + split_foggy_data_file_name[1]
                
                gt_data_file_path = os.path.join(gt_dataset_path, data_file_rel_path_prefix, gt_data_file_name)
                mask_data_file_path = os.path.join(mask_dataset_path, data_file_rel_path_prefix, mask_data_file_name)

                if not os.path.isfile(gt_data_file_path):
                    logging.warn("{} not found | {}\n".format(gt_data_file_name, gt_data_file_path))
                    continue

                if not os.path.isfile(mask_data_file_path):
                    logging.warn("{} not found | {}\n".format(mask_data_file_name, mask_data_file_path))
                    continue

                self.fog_paths.append(foggy_data_file_path)
                self.gt_paths.append(gt_data_file_path)
                self.mask_paths.append(mask_data_file_path)
                
                logging.info("FOGGY FILE: {} | {}\nGT FILE: {} | {}\nMASK FILE: {} | {}\n\n".format(foggy_data_file_name, foggy_data_file_path, gt_data_file_name, gt_data_file_path, mask_data_file_name, mask_data_file_path))
                
                if split == 'val' and one_val_per_dataset:
                    logging.info("User specified to only include one datapoint per dataset. Moving onto next dataset")
                    break

                if dataset == "cityscapes" and cap_cityscapes and num_of_files_added_from_cur_dataset > 250:
                    logging.info("Capping cityscapes at 250")
                    break

                if abs_dataset_cap and num_of_files_added_from_cur_dataset > abs_dataset_cap:
                    logging.info("Capping {} at {}".format(dataset, abs_dataset_cap))
                    break

                num_of_files_added_from_cur_dataset += 1

        logging.info("########  END OF DATA LOADER INIT  ########\n\n")


    def __getitem__(self, index):
        fog_img = Image.open(self.fog_paths[index])
        fog_img = self.img_transform(fog_img.convert('RGB'))
        
        gt_img = Image.open(self.gt_paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[index])
        mask = self.mask_transform(mask.convert('RGB'))

        # uncomment to sanity check that the fog, gt, and mask files correspond to eachother
        '''
        logging.info("INDEX: {}\nFOG_IMG: {}\nGT_IMG: {}\nMASK: {}\n\n".format(index, 
                                                                            self.fog_paths[index].split("/")[-1], 
                                                                            self.gt_paths[index].split("/")[-1], 
                                                                            self.mask_paths[index].split("/")[-1]))
        '''
        
        return fog_img, mask, gt_img

    def __len__(self):
        return len(self.fog_paths)
