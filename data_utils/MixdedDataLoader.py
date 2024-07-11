import numpy as np
import os
import random
from torch.utils.data import Dataset

class MixedBridgePartDataset_bridgenetv1(Dataset):
    """
        This class will read the data from the saved file and give back with the format that can be
        directly used for training. Parapet and pavement are completely removed from this new created
        dataset loader.
    """
    def __init__(self, syn_root, total_num, 
                 n_points, syn_btype_ratio, 
                 data_mode,
                 normalize=True):
        '''
        Args:
            syn_root:
            tatal_num: total number of the training data set
            syn_btype_ratio: numver of straight synthetic bridges
            normalize:
            train:
        '''
        self.syn_root = syn_root
        self.syn_root = os.path.join(self.syn_root, data_mode)
        self.data_mode = data_mode
        self.syn_btype_ratio = syn_btype_ratio
        self.n_points = n_points
        self.syn_num = total_num

        # append the root of all dataset into path list
        #(1) synthetic dataset
        self.straight_root = os.path.join(self.syn_root, "straight_bridge")
        straight_file = os.listdir(self.straight_root)      # the file name of all straight bridge
        self.file_ns = straight_file
        # self.skewed_root   = os.path.join(self.syn_root, "skewed_bridge")
        # skewed_file   = os.listdir(self.skewed_root)        # the file name of all skewd bridge
        self.curved_root   = os.path.join(self.syn_root, "curved_bridge")
        curved_file   = os.listdir(self.curved_root)        # the file name of all curved bridge
        self.file_ns = straight_file + curved_file          # the name of all bridge files

        self.syn_straight_datapath = []
        self.syn_curved_datapath = []
        for file_n in self.file_ns:
            if file_n.split("_")[0] == "straight":
                self.syn_straight_datapath.append(os.path.join(self.straight_root, file_n))
            elif file_n.split("_")[0] == "curved":
                self.syn_curved_datapath.append(os.path.join(self.curved_root, file_n))
            # elif file_n.split("_")[0] == "skewed":
            #    self.syn_datapath.append(os.path.join(self.skewed_root, file_n))
        assert len(self.syn_straight_datapath + self.syn_curved_datapath) > self.syn_num

        # Give the proper ratio between straight and curved bridges to cater to the actual case for the real dataset
        self.syn_straight_num = int(self.syn_num * self.syn_btype_ratio)
        self.syn_curved_num   = int(self.syn_num - self.syn_straight_num)
        self.syn_datapath = random.sample(self.syn_straight_datapath, self.syn_straight_num) + \
                            random.sample(self.syn_curved_datapath, self.syn_curved_num)

        self.datapath = self.syn_datapath
        self.normalize = normalize                      # indicator to whether normalize the point cloud or not
        self.cache = {}
        self.cache_size = 20000

    def pc_normalize(self, pc_unnorm):
        """
            This function will return the normalized point cloud with respect to the mean of the points(-1 to 1)
            Args:
                pc: point cloud

            Returns:
                pc: normalized point cloud
        """
        pc = pc_unnorm.copy()
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def extractor(self, f_path):
        fns = os.listdir(f_path)
        for fn in fns:
            fn_path = os.path.join(f_path, fn)
            fn, _ = os.path.splitext(fn)
            fn_splited = fn.split('_')
            if fn_splited[0] == 'points':
                if len(fn_splited) == 1:
                    xyzs = np.load(fn_path)            # point cloud (x, y, z)
                    xyz = xyzs['xyz'][:, 0:3]
                    xyzs = xyz
                else:
                    label = np.load(fn_path)      # include semantic and instant label
        return xyzs, label

    def __getitem__(self, index):
        if index in self.cache:
            xyzs, label, dtype = self.cache[index]
        else:
            f_path = self.datapath[index]       # return file name given the index number
            _, file_n = os.path.split(f_path)
            if file_n.split('_')[0] in ['straight', 'skewed', 'curved']:
                d_type = 'syn'
                xyzs, label = self.extractor(f_path)
            if len(self.cache) < self.cache_size:
                self.cache[index] = [xyzs, label, d_type]

        if self.normalize:
            xyzs = self.pc_normalize(xyzs)
        if self.data_mode=='train':
            if d_type == 'syn':
                # polute the data with Gaussian distribution
                sigma_1 = np.random.normal(0, 0.005, xyzs.shape[0]).reshape(-1, 1)
                sigma_2 = np.random.normal(0, 0.005, xyzs.shape[0]).reshape(-1, 1)
                sigma_3 = np.random.normal(0, 0.005, xyzs.shape[0]).reshape(-1, 1)
                sigma = np.concatenate([sigma_1, sigma_2, sigma_3], axis=1)
                xyzs[:, 0:3] = xyzs[:, 0:3] + sigma

        sem_label = label['sem_label']
        if self.n_points:   # points need to be subsampled
            sample_indexes = np.random.choice(xyzs.shape[0], self.n_points, replace=False)
            xyzs = xyzs[sample_indexes, :]
            sem_label = sem_label[sample_indexes]
        return xyzs, sem_label

    def __len__(self):
        return len(self.datapath)