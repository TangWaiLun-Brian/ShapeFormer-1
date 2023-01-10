import scipy.io as sio
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(1, "/data/brian22/ShapeFormer")
from xgutils import *
from xgutils.ptutil import make_ply_file_from_pc
import scipy
import h5py
import time
import os
import trimesh
import math
import mcubes
import random
import json
def align_to_shapenet(pc):
    '''
    center = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2
    bounding_box = np.max(pc, axis=0) - np.min(pc, axis=0)
    pc = (pc - center) / np.sqrt(np.sum(bounding_box**2)) * 1
    '''
    angle = math.pi
    rotation_matrix = np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)]])
    pc = np.matmul(rotation_matrix, pc.T).T

    return pc
class Imnet2Dataset(Dataset):
    def __init__(self, dataset='IMNet2_packed', cate="all", zoomfac=1,
                 duplicate_size=1, split='train', boundary_N=2048, target_N=-1, grid_dim=256, weighted_sampling=False, Xbd_as_Xct=False, Xct_as_Xbd=False,
                 partial_opt={"class": "shapeformer.data.partial.BallSelector",
                               "kwargs": dict(radius=.4, context_N=512)}):
        self.__dict__.update(locals())
        self.split = split
        self.grid_dim = grid_dim

        self.dpath = dpath = f'/data/brian22/ShapeFormer/datasets/IMNet2_packed/{self.split}.hdf5'#f'datasets/{dataset}/{split}.hdf5'
        self.weighted_sampling = weighted_sampling
        with h5py.File(dpath, 'r') as f:
            total_length = f['Xbd'].shape[0]
            all_ind = np.arange(total_length)
            if type(cate) is str:
                if cate == "all":
                    self.subset = all_ind
                else:
                    self.subset = np.array(f[f"cate_{cate}"])
            elif type(cate) is list:
                self.subset = np.concatenate(
                    [np.array(f[f"cate_{cat}"]) for cat in cate])
                print("subset num", len(self.subset))
        self.length = len(self.subset)
        if split != "train":
            self.duplicate_size = 1
        else:
            self.duplicate_size = duplicate_size
        #self.split = split
        self.boundary_N, self.target_N = boundary_N, target_N
        self.partial_opt = partial_opt
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

        self.Xbds = nputil.H5Var(self.dpath, "Xbd")
        self.shape_vocabs = nputil.H5Var(self.dpath, "shape_vocab")
        self.vocab_idxs = nputil.H5Var(self.dpath, "vocab_idx")

        self.all_Xtg = nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [grid_dim, ]*3, indexing="ij")

        #imnet_datapath =    "datasets/IM-NET"
        imnet_path = "/data/brian22/ShapeFormer/datasets/IMNet2_packed/"#os.path.join(imnet_datapath, "IMSVR/data")
        self.imnet_fixed_path = os.path.join("/data/brian22/ShapeFormer/datasets/IMNet2_fixed_data/", self.split)
        #print(self.imnet_fixed_path)
        lines = open(imnet_path + f"/shape_name_{self.split}.txt", "r").readlines()
        self.shape_names = [line.strip() for line in lines ]#if '03001627' in line

    def __len__(self):
        return self.length * self.duplicate_size

    def __getitem__(self, index, all_target=False):
        index = index % self.length
        o_ind = index
        index = self.subset[index]
        with h5py.File(self.dpath, 'r') as f:
            Xbd = trimesh.load(os.path.join(self.imnet_fixed_path, self.shape_names[index], 'gt.ply'))
            Xbd = Xbd.vertices.astype(np.float32)
            Xbd = torch.from_numpy(Xbd).cpu().numpy()
            #Xbd = self.Xbds[index]
            Xbd = align_to_shapenet(Xbd)

            #Xct = np.float32(self.get_partial(Xbd, o_ind))
            Xct = trimesh.load(os.path.join(self.imnet_fixed_path, self.shape_names[index], 'partial.ply'))
            Xct = torch.from_numpy(Xct.vertices.astype(np.float32)).cpu().numpy()
            Xct = align_to_shapenet(Xct)

            choice = np.random.choice(
                Xbd.shape[0], self.boundary_N, replace=True)
            Xbd = Xbd[choice]
            shape_vocab, vocab_idx = self.shape_vocabs[index], self.vocab_idxs[index]
            Xtg, Ytg = self.get_target(
                shape_vocab, vocab_idx, all_target=all_target)
            #print(Xtg.shape, Ytg.shape)
            if self.Xct_as_Xbd == True:
                Xbd = Xct
            #print("Xtg", Xtg.shape, Ytg.shape)
            #print(index, o_ind)
            #print(self.shape_names[index])
            #print(self.shape_names[o_ind])
            #test_path = "/data/brian22/ShapeFormer/experiments/shapeformer/shapenet_scale/results/"
            #vertices, traingles = mcubes.marching_cubes(Ytg.reshape(64, 64, 64), 0.0)
            #vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
            #vertices = vertices * 0.9

            #mcubes.export_obj(vertices, traingles, os.path.join(test_path, f'Xtg_{index}.obj'))
            item = dict(Xct=Xct,
                        Xbd=Xbd,
                        Xtg=torch.from_numpy(Xtg),
                        Ytg=torch.from_numpy(Ytg),
                        index=torch.from_numpy(np.array(index)).long(),
                        )
            #print(item)
            #item = ptutil.nps2ths(item)
            item = ptutil.ths2nps(item)
            '''
            P = f"/data/ssd/brian22/summer_intern/others_work/ShapeFormer/datasets/IMNet2_fixed_data/train/"#{self.split}
            P = os.path.join(P, self.shape_names[index])
            print(P)
            cmd = f"mkdir {P}"
            os.system(cmd)
            make_ply_file_from_pc(os.path.join(P, f'gt.ply'), item['Xbd'])
            make_ply_file_from_pc(os.path.join(P, f'partial.ply'), item['Xct'])
            '''
            #print(item)

        return item

    def get_partial(self, Xbd, Xtg=None, Ytg=None, index=None):
        Xct = self.partial_selector(Xbd, index=index)
        return Xct

    def get_target(self, shape_vocab, vocab_idx, all_target=False):
        voxels = ptutil.decompress_voxels(shape_vocab, vocab_idx)
        #print('voxel size: ', voxels.shape)
        folded = voxels[:,:,::4] + voxels[:,:,1::4] + voxels[:,:,2::4] + voxels[:,:,3::4]
        folded = folded[:, ::4, :] + folded[:, 1::4, :] + folded[:, 2::4, :] + folded[:, 3::4, :]
        folded = folded[::4, :, :] + folded[1::4, :, :] + folded[2::4, :, :] + folded[3::4, :, :]
        voxels = np.clip(folded, 0, 1)
        #print('down-sampled voxel size: ', voxels.shape)
        x_dim, grid_dim = len(voxels.shape), voxels.shape[-1]
        if self.target_N == -1 or all_target == True:
            Xtg = self.all_Xtg
            Ytg = voxels.reshape(-1, 1)
        else:
            if self.weighted_sampling == True:
                #print('weight sampling !!!!!!!!!!')
                rdind_uniform = torch.randint(
                    0, grid_dim, (self.target_N//2, x_dim))
                flat = voxel.reshape(-1)
                inside_pos = np.where(flat)[0]
                outside_pos = np.where(flat)[0]
                rdc1 = np.random.choice(len(inside_pos),  self.target_N//2)
                rdc2 = np.random.choice(len(outside_pos), self.target_N//2)
                choice = np.concatenate([inside_pos[rdc1], outside_pos[rdc2]])
                inds = ptutil.unravel_index(
                    torch.from_numpy(choice), shape=(256, 256, 256))
            else:
                inds = torch.randint(0, grid_dim, (self.target_N, x_dim))
                #print('inds size: ', inds.size())
            Xtg = ptutil.index2point(inds, grid_dim=grid_dim).numpy()
            Ytg = voxels[inds[:, 0], inds[:, 1], inds[:, 2]][..., None]
            # allind = ptutil.unravel_index(torch.arange(grid_dim**3), (256,256,256))
            # Xtg = ptutil.index2point(allind, grid_dim=grid_dim).numpy()
            # Ytg = voxels[allind[:,0], allind[:,1], allind[:,2]][...,None]
        return Xtg, Ytg

    @classmethod
    def unittest(cls, **kwargs):
        train_dset = cls(boundary_N=102400, target_N=8192, **kwargs)
        ts = []
        ts.append(time.time())
        ditems = [train_dset[i] for i in range(32)]
        ts.append(time.time())
        print("training dataloading time", ts[-1]-ts[-2])

        dset = cls(boundary_N=102400, **kwargs)
        ts.append(time.time())
        ditems = [dset[i] for i in range(1)]
        ts.append(time.time())
        print("dataloading time", ts[-1]-ts[-2])
        ditem = ditems[0]
        voxelized = ptutil.point2voxel(torch.from_numpy(ditem["Xbd"])[
                                       None, ...], grid_dim=64).reshape(-1).numpy()
        Xtg64 = nputil.makeGrid([-1, -1, -1], [1, 1, 1], [64, 64, 64])
        img = npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=Xtg64, Ytg=voxelized, pred_y=ditem["Ytg"],
                                    show_images=["GT"])[1][0]
        visutil.showImg(img)

        imgs = [npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=ditem["Xtg"], Ytg=ditem["Ytg"], pred_y=ditem["Ytg"],
                show_images=["GT"])[1][0] for ditem in ditems]
        ts.append(time.time())
        print("plot time", ts[-1]-ts[-2])
        print(imgs[0].shape)
        grid = visutil.imageGrid(imgs, zoomfac=1)
        visutil.showImg(grid)
        return dset, ditems, imgs, grid


class Imnet2LowResDataset(Dataset):
    def __init__(self, dataset='IMNet2_64', cate="all", zoomfac=1,
                 duplicate_size=1, split='train', boundary_N=2048, target_N=-1, grid_dim=64, weighted_sampling=False, Xbd_as_Xct=False, Xct_as_Xbd=False,
                 partial_opt={"class": "shapeformer.data.partial.BallSelector",
                               "kwargs": dict(radius=.4, context_N=512)}):
        self.__dict__.update(locals())
        self.split = split
        
        is_truncate = False
        if self.split == "val":
            is_truncate = True
            self.split = "train"
        
        self.dpath = dpath = f'/data/brian22/ShapeFormer/datasets/{dataset}/{self.split}.hdf5'#f'datasets/{dataset}/{split}.hdf5'
        self.weighted_sampling = weighted_sampling
        self.grid_dim = grid_dim
        with h5py.File(dpath, 'r') as f:
            total_length = f['Xbd'].shape[0]
            all_ind = np.arange(total_length)
            if type(cate) is str:
                if cate == "all":
                    self.subset = all_ind
                else:
                    self.subset = np.array(f[f"cate_{cate}"])
            elif type(cate) is list:
                self.subset = np.concatenate(
                    [np.array(f[f"cate_{cat}"]) for cat in cate])
                print("subset num", len(self.subset))
        #self.subset = self.subset[::2000]

        # if is_truncate:
        #     self.subset = self.subset[:len(self.subset)//10]
        # elif self.split == "train":
        #     self.subset = self.subset[len(self.subset) // 10:]

        #self.subset = self.subset[:1]
        self.length = len(self.subset)
        #print('LENGTH:', self.length)
        if split != "train":
            self.duplicate_size = 1
        else:
            self.duplicate_size = duplicate_size
        self.split = split
        self.boundary_N, self.target_N = boundary_N, target_N
        self.partial_opt = partial_opt
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

        self.Xbds = nputil.H5Var(self.dpath, "Xbd")
        self.Ytgs = nputil.H5Var(self.dpath, "Ytg")  # packed

        self.all_Xtg = nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [grid_dim, ]*3, indexing="ij")
        
        # imnet_datapath =    "datasets/IM-NET"
        imnet_path = "/data/brian22/ShapeFormer/datasets/IMNet2_packed/"  # os.path.join(imnet_datapath, "IMSVR/data")
        self.imnet_fixed_path = os.path.join("/data/brian22/ShapeFormer/datasets/IMNet2_fixed_data/", self.split)
        # print(self.imnet_fixed_path)
        lines = open(imnet_path + f"/shape_name_{self.split}.txt", "r").readlines()
        self.shape_names = [line.strip() for line in lines]
        self.args = random.Random(1234)

    def __len__(self):
        return self.length * self.duplicate_size

    def __getitem__(self, index, all_target=False):
        index = index % self.length
        o_ind = index
        index = self.subset[index]
        #print(index, o_ind)
        #name = self.shape_names[o_ind]
        with h5py.File(self.dpath, 'r') as f:
            #Xbd = self.Xbds[index]  #index
            #Xct = np.float32(self.get_partial(Xbd, index=o_ind))
            if self.split == 'test':
                rand_index = 0
            else:
                rand_index = self.args.randint(0, 20-1)
            Xbd = trimesh.load(os.path.join(self.imnet_fixed_path, self.shape_names[index], f'gt.ply'))
            Xbd = torch.from_numpy(Xbd.vertices).cpu().numpy()
            Xct = trimesh.load(os.path.join(self.imnet_fixed_path, self.shape_names[index], f'partial_{rand_index}.ply'))
            Xct = torch.from_numpy(Xct.vertices).cpu().numpy()
            #l, s= np.max(Xbd, axis=0), np.min(Xbd, axis=0)
            #print('max:', np.max(l), 'min:', np.min(s))
            #print('length:', l-s)
            choice = np.random.choice(Xbd.shape[0], self.boundary_N, replace=True)
            Xbd = Xbd[choice]
            choice = np.random.choice(Xct.shape[0], self.partial_opt["kwargs"]["context_N"], replace=True)
            Xct = Xct[choice]
            
            x_dim = Xbd.shape[-1]
            Ytg = np.unpackbits(self.Ytgs[index], axis=-1)[..., None]   #index
            Xtg = self.all_Xtg
            #print(Xtg.shape, Ytg.shape)
            if self.weighted_sampling == True:
                target_N = self.target_N if self.target_N != - \
                    1 else Xtg.shape[0]
                Xtg, Ytg = balanced_sampling2(
                    Xbd, Xtg, Ytg, target_N=target_N, x_dim=x_dim, grid_dim=self.grid_dim)
            else:
                if self.target_N != -1 and all_target == False:
                    choice = np.random.choice(
                        Xtg.shape[0], self.target_N, replace=True)
                    Xtg = Xtg[choice]
                    Ytg = Ytg[choice]
            if self.Xct_as_Xbd == True:
                Xbd = Xct
            #print(Xtg.shape, Ytg.shape)
            #print(index, o_ind)
            #print(self.shape_names[index])
            #print(self.shape_names[o_ind])
            #tmp_dir = "/data/brian22/ShapeFormer/experiments/"
            #make_ply_file_from_pc(os.path.join(tmp_dir, 'test.ply'), Xct.astype(np.float32))


            # P = f"/data/brian22/ShapeFormer/datasets/IMNet2_fixed_data/train/"  # {self.split}
            # P = os.path.join(P, self.shape_names[index])
            # print(P)
            # cmd = f"mkdir {P}"
            # os.system(cmd)
            # make_ply_file_from_pc(os.path.join(P, f'gt.ply'), Xbd)
            # for i in range(20):
            #     Xct = np.float32(self.get_partial(Xbd, index=o_ind))
            #     make_ply_file_from_pc(os.path.join(P, f'partial_{i}.ply'), Xct)
            item = dict(Xct=Xct.astype(np.float32),
                        Xbd=Xbd.astype(np.float32),
                        Xtg=Xtg.astype(np.float32),
                        Ytg=Ytg.astype(np.float32),
                        index=np.array(index).astype(np.intc),
                        )
            #item = ptutil.nps2ths(item,)
        return item

    def get_partial(self, Xbd, Xtg=None, Ytg=None, index=None):
        if self.Xbd_as_Xct == True:
            return Xbd
        Xct = self.partial_selector(Xbd, index=index)
        return Xct

    @classmethod
    def unittest(cls, grid_dim=32, **kwargs):
        train_dset = cls(grid_dim=grid_dim, boundary_N=102400,
                         target_N=8192, **kwargs)
        ts = []
        ts.append(time.time())
        ditems = [train_dset[i] for i in range(32)]
        ts.append(time.time())
        print("training dataloading time", ts[-1]-ts[-2])

        boundary_N = 4096
        dset = cls(boundary_N=boundary_N, **kwargs)
        ts.append(time.time())
        ditems = [dset[i] for i in range(1)]
        ts.append(time.time())
        print("dataloading time", ts[-1]-ts[-2])
        ditem = ditems[0]
        print(f"point res={256} {boundary_N}, grid_dim={64} point2voxel")
        voxelized = ptutil.point2voxel(torch.from_numpy(ditem["Xbd"])[
                                       None, ...], grid_dim=64).reshape(-1).numpy()
        Xtg64 = nputil.makeGrid([-1, -1, -1], [1, 1, 1], [64, 64, 64])
        img = npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=Xtg64, Ytg=voxelized, pred_y=ditem["Ytg"],
                                    show_images=["GT"])[1][0]
        visutil.showImg(img)

        print(f"ground truth grid_dim={grid_dim}")
        imgs = [npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=ditem["Xtg"], Ytg=ditem["Ytg"], pred_y=ditem["Ytg"],
                show_images=["GT"])[1][0] for ditem in ditems]
        ts.append(time.time())
        print("plot time", ts[-1]-ts[-2])
        print(imgs[0].shape)
        grid = visutil.imageGrid(imgs, zoomfac=1)
        visutil.showImg(grid)
        return dset, ditems, imgs, grid

# def vis_imnet(dset, vis_dir):
#     vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
#         camUp=np.array([0,1,0]),camHeight=2,resolution=(256,256), samples=16)
#     for i in range(len(dset)):
#         item = dset[i]
#         item["Xct"]
#         item["Xbd"]
#         item["Xtg"]
#         item["Ytg"]


def balanced_sampling(Xbd, Xtg, Ytg, target_N=4096, x_dim=3, grid_dim=32):
    rdind_uniform = torch.randint(0, grid_dim, (target_N//2, x_dim))
    inside_pos = np.where(Ytg)[0]
    outside_pos = np.where(1-Ytg)[0]
    rdc_xbd = np.random.choice(Xbd.shape[0], target_N//2, replace=True)
    sub_Xbd = Xbd[rdc_xbd]

    rdc1 = np.random.choice(len(inside_pos),  target_N//4, replace=True)
    rdc2 = np.random.choice(len(outside_pos), target_N//4, replace=True)
    choice = np.concatenate([rdc_xbd, inside_pos[rdc1], outside_pos[rdc2]])
    #inds = ptutil.unravel_index(torch.from_numpy(choice), shape=(256,256,256))
    sub_Xtg = np.concatenate([Xtg[choice], sub_Xbd])
    sub_Ytg = np.concatenate([Ytg[choice], np.zeros((sub_Xbd.shape[0], 1))+.5])
    return sub_Xtg, sub_Ytg


def balanced_sampling2(Xbd, Xtg, Ytg, target_N=4096, x_dim=3, grid_dim=32, random_scale=.1):
    rdind_uniform = torch.randint(0, grid_dim, (target_N//2, x_dim))
    rdc_xbd = np.random.choice(Xbd.shape[0], target_N//2, replace=True)
    sub_Xbd = Xbd[rdc_xbd] + np.random.randn(len(rdc_xbd), x_dim)*random_scale
    sub_Xbd_ind = ptutil.point2index(torch.from_numpy(
        sub_Xbd), grid_dim=grid_dim, ravel=True).numpy()
    sub_Xbd_Y = Ytg[sub_Xbd_ind]

    rdc1 = np.random.choice(Xtg.shape[0],  target_N//2, replace=True)
    choice = np.concatenate([rdc_xbd, rdc1])
    #inds = ptutil.unravel_index(torch.from_numpy(choice), shape=(256,256,256))
    sub_Xtg = Xtg[choice]
    sub_Ytg = Ytg[choice]
    #sub_Xtg = np.concatenate([Xtg[choice], sub_Xbd])
    #sub_Ytg = np.concatenate([Ytg[choice], sub_Xbd_Y])
    return sub_Xtg, sub_Ytg


def generate_dataitem(shape_path):
    shape_name = shape_path[1]
    shape_path = shape_path[0]
    loaded = sio.loadmat(shape_path)  # , allow_pickle=True)
    shape_vocab, vocab_idx = loaded["b"].reshape(
        loaded["b"].shape[0], -1), (loaded["bi"]-1).astype(int).reshape(-1)
    folded = ptutil.decompress_voxels(shape_vocab, vocab_idx, unpackbits=False)
    folded = geoutil.shapenetv1_to_shapenetv2(folded)
    folded = geoutil.shapenetv2_to_cart(folded)

    #print('Move onto array2mesh')
    v256, f256 = geoutil.array2mesh(folded.reshape(-1), dim=3, bbox=np.array(
        [[-1, -1, -1], [1, 1, 1.]])*1., thresh=.5, if_decimate=False, cart_coord=True)
    #print('Move onto sample points from mesh')
    Xbd = geoutil.sampleMesh(v256, f256, 65536)
    #Xbd = Xbd.astype(np.float32) / 2
    #Xbd = align_to_shapenet(Xbd)

    folded = folded[:, :, ::4] + folded[:, :, 1::4] + folded[:, :, 2::4] + folded[:, :, 3::4]
    folded = folded[:, ::4, :] + folded[:, 1::4, :] + folded[:, 2::4, :] + folded[:, 3::4, :]
    folded = folded[::4, :, :] + folded[1::4, :, :] + folded[2::4, :, :] + folded[3::4, :, :]
    folded = np.clip(folded, 0, 1)
    # print(folded.shape)

    # shape_vocab, vocab_idx = ptutil.compress_voxels(folded, packbits=True)
    bits = np.packbits(folded.astype(bool))
    
    #dir = "/data/brian22/ShapeFormer/experiments/"
    #make_ply_file_from_pc(os.path.join(dir, shape_name.replace('/', '-')+'.ply'), Xbd)
    return  bits, Xbd, shape_name #shape_vocab, vocab_idx,


def make_imnet_dataset():
    imnet_datapath = "/data/brian22/ShapeFormer/datasets/IM-NET/"#"datasets/IM-NET"
    hspnet_datapath = "/data/brian22/ShapeFormer/datasets/shapenet/"#"datasets/hsp_shapenet"
    imnet_path = os.path.join(imnet_datapath, "IMSVR/data")
    hspnet_path = os.path.join(hspnet_datapath, "modelBlockedVoxels256")
    train_shapeh5 = imnet_path + "/all_vox256_img_train.hdf5"
    test_shapeh5 = imnet_path + "/all_vox256_img_test.hdf5"
    split_root = '/data/brian22/ShapeFormer/datasets/PartNet_Split/'
    use_cate = ['03001627', '04379243']

    # IMNET split dir
    # lines = open(imnet_path+"/all_vox256_img_train.txt", "r").readlines()
    # train_shape_names = [line.strip() for line in lines]
    # lines = open(imnet_path+"/all_vox256_img_test.txt", "r").readlines()
    # test_shape_names = [line.strip() for line in lines]

    # PartNet split dir
    train_shape_names = []
    test_shape_names = []
    for cate in use_cate:
        with open(os.path.join(split_root, f'partnet_{cate}_train.json'), 'r') as f:
            info = json.load(f)
        lines = [cate+'/'+item['model_id'] for item in info]
        train_shape_names += lines

        with open(os.path.join(split_root, f'partnet_{cate}_test.json'), 'r') as f:
            info = json.load(f)
        lines = [cate+'/'+item['model_id'] for item in info]
        test_shape_names += lines

    target_dir = "/data/brian22/ShapeFormer/datasets/PartNet_packed/"#"datasets/IMNet2_packed/"
    sysutil.mkdirs(target_dir)


    typelist = [train_shape_name.split("/")[0]
                for train_shape_name in train_shape_names if train_shape_name.split("/")[0] in use_cate]
    unique_types = np.unique(typelist)
    type_dict = dict([(typ, i) for i, typ in enumerate(unique_types)])

    cates = [[] for typ in type_dict]
    start_idx = 0
    for si in range(len(train_shape_names)):
        shape_name = train_shape_names[si]
        if shape_name.split("/")[0] in type_dict:
            type_ind = type_dict[shape_name.split("/")[0]]
            cates[type_ind].append(start_idx) #cates[type_ind].append(si)
            start_idx += 1
    shape_paths = [(hspnet_path+"/"+shape_name +
                   ".mat", shape_name) for shape_name in train_shape_names if shape_name.split("/")[0] in type_dict]
    #print(len(shape_paths))
    #shape_paths = shape_paths[::2000]
    bits, Xbds, shape_names = sysutil.parallelMap(
        generate_dataitem, [shape_paths], zippedIn=False)   #shape_vocabs, vocab_idxs,

    dataDict = {"Ytg": np.array(bits), "Xbd": np.array(Xbds)}
    name_path = "/data/brian22/ShapeFormer/datasets/PartNet_packed/shape_name_train.txt"
    with open(name_path, 'w') as f:
        for n in shape_names:
            f.write(f"{n}\n")
    for ci in range(len(cates)):
        dataDict[f"cate_{ci}"] = np.array(cates[ci])
    nputil.writeh5(target_dir+"train.hdf5", dataDict)

    cates = [[] for typ in type_dict]
    start_idx = 0
    for si in range(len(test_shape_names)):
        shape_name = test_shape_names[si]
        if shape_name.split("/")[0] in type_dict:
            type_ind = type_dict[shape_name.split("/")[0]]
            cates[type_ind].append(start_idx)
            start_idx += 1

    shape_paths = [(hspnet_path+"/"+shape_name +
                   ".mat", shape_name) for shape_name in test_shape_names if shape_name.split("/")[0] in type_dict]
    #print(len(shape_paths))
    #shape_paths = shape_paths[::2000]
    bits, Xbds, shape_names = sysutil.parallelMap(
        generate_dataitem, [shape_paths], zippedIn=False)
    dataDict = {"Ytg": np.array(bits), "Xbd": np.array(Xbds)}
    name_path = "/data/brian22/ShapeFormer/datasets/PartNet_packed/shape_name_test.txt"
    with open(name_path, 'w') as f:
        for n in shape_names:
            f.write(f"{n}\n")
    for ci in range(len(cates)):
        dataDict[f"cate_{ci}"] = np.array(cates[ci])
    nputil.writeh5(target_dir+"test.hdf5", dataDict)


def IMNet2_h5_unittest():
    dataDict = nputil.readh5("datasets/IMNet2/test.h5")
    print(dataDict.keys())
    print([dataDict["shape_vocab"][i].shape for i in range(10)])
    a, b, c = dataDict["shape_vocab"][0], dataDict["vocab_idx"][0], dataDict["Xbd"][0]
    #a,b,c = generate_dataitem("datasets/hsp_shapenet/modelBlockedVoxels256/04530566/cc3957e0605cd684bb48c7922d71f3d0.mat")
    unfold = ptutil.decompress_voxels(a, b)
    dflt_camera = dict(camPos=np.array([2, 2, 2]), camLookat=np.array([0., 0., 0.]),
                       camUp=np.array([0, 1, 0]), camHeight=2.414, resolution=(512, 512), samples=256)
    vert, face = geoutil.array2mesh(unfold.reshape(-1), dim=3, bbox=np.array(
        [[-1, -1, -1], [1, 1, 1.]])*1., thresh=.5, if_decimate=False, cart_coord=True)
    print(f'vert shape: {vert.shape}, face shape: {face.shape}')
    gtmesh = fresnelvis.renderMeshCloud(
        mesh={'vert': vert, 'face': face}, cloud=c, cloudR=0.001, axes=True, **dflt_camera)
    visutil.showImg(gtmesh)
# IMNet2_h5_unittest()

if __name__ == '__main__':
    make_imnet_dataset()
