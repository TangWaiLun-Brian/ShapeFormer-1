import logging
import numpy as np
import trimesh
from pykdtree.kdtree import KDTree
from plyfile import PlyData
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scipy.linalg import sqrtm
#from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d
from pointnet import PointNetCls
#from data.dataset_benchmark import BenchmarkDataset
#from im2mesh.utils.io import export_pointcloud

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

EMPTY_PCL_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer": 6,
}

EMPTY_PCL_DICT_NORMALS = {
    "normals completeness": -1.0,
    "normals accuracy": -1.0,
    "normals": -1.0,
}

logger = logging.getLogger(__name__)

print(torch.cuda.is_available())
class MeshEvaluator(object):
    """ Mesh evaluation class.
    It handles the mesh evaluation process.
    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, obj_id, mesh, pointcloud_tgt, normals_tgt, save_dir, thresholds=np.linspace(1.0 / 1000, 1, 1000), center=None, scale=None):
        """ Evaluates a mesh.
        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            thresholds (numpy arry): for F-Score
        """
        if isinstance(mesh, trimesh.PointCloud) and len(mesh.vertices) != 0:
            pointcloud = mesh.vertices
            pointcloud = pointcloud.astype(np.float32)
            if pointcloud.shape[0] > self.n_points:
                index = np.random.choices(list(range(pointcloud.shape[0])), self.n_points, replace=False)
            else:
                index = np.random.choice(list(range(pointcloud.shape[0])), self.n_points, replace=True)
            pointcloud = pointcloud[index,:]
            
            normals = None
        elif len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)

            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(obj_id, pointcloud, pointcloud_tgt, normals, normals_tgt, save_dir,  thresholds=thresholds, center=center, scale=center)

        return out_dict

    def eval_pointcloud(
        self, obj_id, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, save_dir=None, thresholds=np.linspace(1.0 / 1000, 1, 1000), center=None, scale=None
    ):
        """ Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warn("Empty pointcloud / mesh detected!")
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-10) for i in range(len(precision))]

        # FPD-Score
        FPD = calculate_fpd(pointcloud, pointcloud_tgt)
        print('FPD:', FPD)
        

        out_dict = {
            "id":obj_id,
            "center":center,
            "scale":scale, 
            "pp":pointcloud, 
            "tp":pointcloud_tgt, 
            "pn":normals, 
            "tn":normals_tgt,
            "completeness": completeness,
            "accuracy": accuracy,
            "normals completeness": completeness_normals,
            "normals accuracy": accuracy_normals,
            "normals": normals_correctness,
            "completeness2": completeness2,
            "accuracy2": accuracy2,
            "chamfer-L2": chamferL2,
            "chamfer-L1": chamferL1,
            "F":F,
            "f-score": F[9],  # threshold = 1.0%
            "f-score-15": F[14],  # threshold = 1.5%
            "f-score-20": F[19],  # threshold = 2.0%
            "f-score-30": F[29],
            "f-score-40": F[39],
            "f-score-50": F[49],
            "f-score-60": F[59],
            "fpd-score": FPD,
        }
        np.save("./"+ save_dir + "/" + obj_id+".npy",out_dict)

        return out_dict

################################################################################################################################################
"""Calculate Frechet Pointcloud Distance referened by Frechet Inception Distance."
    [ref] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
    github code  : (https://github.com/bioinf-jku/TTUR)
    paper        : (https://arxiv.org/abs/1706.08500)
"""


def get_activations(pointclouds, model, batch_size=100, dims=1808,
                    device=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    n_batches = pointclouds.size(0) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    pointclouds = pointclouds.transpose(1, 2)
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        pointcloud_batch = pointclouds[start:end]

        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)

        _, _, actv = model(pointcloud_batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = actv.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(pointclouds, model, batch_size=100,
                                    dims=1808, device=None, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(pointclouds, model, batch_size, dims, device, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['m'][:], f['s'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def save_statistics(real_pointclouds, path, model, batch_size, dims, cuda):
    m, s = calculate_activation_statistics(real_pointclouds, model, batch_size,
                                           dims, cuda)
    np.savez(path, m=m, s=s)
    print('save done !!!')


def calculate_fpd(pointclouds1, pointclouds2=None, batch_size=1, dims=1808, device=None):
    """Calculates the FPD of two pointclouds"""

    PointNet_path = 'cls_model_39.pth'
    statistic_save_path = './evaluation/pre_statistics.npz'
    model = PointNetCls(k=16)
    model.load_state_dict(torch.load(PointNet_path))

    if device is not None:
        model.to(device)

    m1, s1 = calculate_activation_statistics(torch.from_numpy(pointclouds1).unsqueeze(0), model, batch_size, dims, device)
    if pointclouds2 is not None:
        m2, s2 = calculate_activation_statistics(torch.from_numpy(pointclouds2).unsqueeze(0), model, batch_size, dims, device)
    else:  # Load saved statistics of real pointclouds.
        f = np.load(statistic_save_path)
        m2, s2 = f['m'][:], f['s'][:]
        f.close()

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
##########################################################################################################################################
def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    #print(points_tgt.shape)
    #print(points_src.shape)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """ Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def check_missing_part(exp_name, shape_list,epoch):
    
    missing_part = []
    for shape_item in shape_list:
        exp_base_dir = "/apdcephfs/share_1467498/datasets/runsongzhu/IGR/%s/exps/" % exp_name


        timestamps = os.listdir(os.path.join(exp_base_dir, shape_item))
        timestamp = sorted(timestamps)[-1]
        print(timestamp)

        file_data = "%s/%s/%s/plots/igr_%s_%s.ply" % (exp_base_dir, shape_item, timestamp, epoch, shape_item)
                    
        if not os.path.exists(file_data):
            missing_part.append(shape_item)

    return missing_part

def myMkdir(input_path):
    if not os.path.exists(input_path):
        os.mkdir(input_path)


def generate_dir():
    data2SaveMeshDir = dict()
    data2SaveMeshDir["thingi"] = "scaled_challenging_data/"
    data2SaveMeshDir["abc_noisefree"] = "abc_noisefree"
    all_datasets = ["low_noise", "med_noise", "high_noise"]
    for data_type in all_datasets:
        data2SaveMeshDir[data_type] = "%s/%s"%("noisy", data_type)
    
    all_datasets = ["vardensity_gradient", "vardensity_striped"]
    
    for data_type in all_datasets:
        data2SaveMeshDir[data_type] = "%s/%s"%("density", data_type)

    return data2SaveMeshDir


if __name__ == "__main__":
    
    import glob
    # shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
    # dataset_name = "thingi"
    # exp_name = "things_full_model_by_PCA_normal_SAL_v9"
    # exp_name_2 = "things_full_model_by_PCA_normal_SAL_v10"
    # exp_name_all = ["things_full_model_by_PCA_normal_SAL_v9", "things_full_model_by_PCA_normal_SAL_v10"]
    # exp_name = "things_full_model_by_PCA_normal_geometry_prior_data_80_SAL_1"
    # data_type = ["abc_noisefree", "density_variation", "noisy_data"]
    data_type = ["thingi"]
    # data_type = ["density_variation"]
    epoch = 20000 # DEFAULT: 20,000
    # methods_name = ["SSN_fitting_ablantion_v1","ablation_balance", "ablation_balance_v2"]
    # methods_name = [ "SAP", "NeuralPull", "IGR", "PSR", "SAL","SALD", "DMLS", "run_SSN_Fitting_code_v8_run_SSN_Fitting_thinigi_new_1_v1"]
    # methods_name = ["SAP", "SALD", "run_SSN_Fitting_code_v8_run_SSN_Fitting_thinigi_new_1_v1"]
    # methods_name = ["SAP", "SALD", "run_SSN_Fitting_code_v8_run_SSN_Fitting_thinigi_new_1_v1"]
    #base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/"
    #base_dir = "/data3/brian22/occupancy_networks/data/IMNet_mount/IMNet2_fixed_data"
    base_dir = "/data/brian22/ShapeFormer/datasets/IMNet2_fixed_data/"
    #shape_name_dir = "/data3/brian22/occupancy_networks/data/shape_name_test.txt"
    shape_name_dir = "/data/brian22/ShapeFormer/datasets/IMNet2_packed/shape_name_test.txt"
    #all_datatype_dir = generate_dir()
    # methods_name=["Our", "SAP","NP", "DeepMLS"]
    # methods_name=["derivative_ablation", "signed_ablation"]
    # methods_name=["Our",  "compare_traditional_sample"]
    #############################################################################################################################
    methods_name = ["ShapeFormer"]
    dataset = 'shapenet'
    shape_list = ["04379243"]#["03001627"]
    # base_dir2 = "/data3/brian22/PoinTr/experiments/PoinTr/ShapeNet55_models/04379243"
    # base_dir2 = "/data3/brian22/occupancy_networks/out/shapenet_chair/"
    # base_dir2 = "/data3/brian22/SA-ConvONet/out/shapenet_chair/"
    # base_dir2 = "/data3/brian22/PoinTr/experiments/PoinTr/ShapeNet55_models/test_IMNET_Chair/"
    base_dir2 = "/data/brian22/ShapeFormer/experiments/shapeformer/shapenet_scale/results/VisShapeFormer/eval_shapenet_table/"
    num_pts = 8192
    #############################################################################################################################
    # print(methods_name[-1])
    # methods_name = ["SSN_fitting_ablantion_v8_all_v16","SSN_fitting_ablantion_v8_all_v8","SSN_fitting_ablantion_v8_all_v1", "SSN_fitting_ablantion_v8_all", "run_SSN_Fitting_v8_all_focal", "run_SSN_Fitting_v8_all_focal_v1"]
    # methods_name = ["ablation_balance_11", "ablation_balance_v2", "ablation_importance", "ablation_signed_module"]
    gt_cache={}
    n_points = 50000
    evaluator = MeshEvaluator(n_points=n_points)
    device = "cpu"
    

    print("good!! and  check")
    metric_save_dir = "metric_main_compares"
    if not os.path.exists(metric_save_dir):
        os.mkdir(metric_save_dir)
    


        

    # epoch = 20000
    # for type_id, data_type in enumerate(all_types):

    for idx, type_item in enumerate(data_type):
        
        # exp_name_list = ["all_adaptive_grid_v6_v5_new_sample_v3_fix_bug_v9_v1_clean_v22_log_v1_abc_noisefree_adaptive_weight_v3/%s/"%(type_item),
        #     "/apdcephfs/share_1467498/datasets/runsongzhu/sap_final/exps/%s/"%(type_item),
        #     "/apdcephfs/share_1467498/datasets/runsongzhu/NeuralPull/pre_model/%s_sur"%(type_item),
        # ]
        # file_name = "/apdcephfs/share_1467498/home/runsongzhu/IGR/data/optim_data/%s_scale_1/testset.txt"%(type_item)
        
        # with open(file_name, 'r') as f:
        #     shape_list = f.readlines()
        # shape_list = [file_name.strip() for file_name in shape_list]
        # shape_list = shape_list
        #shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
        #shape_list = ["03001627"]

        compare_num = len(methods_name)
        fpd = [[] for i in range(compare_num)]
        f_10 = [[] for i in range(compare_num)]
        f_15 = [[] for i in range(compare_num)]
        f_20 = [[] for i in range(compare_num)]
        f_30 = [[] for i in range(compare_num)]
        f_40 = [[] for i in range(compare_num)]
        f_50 = [[] for i in range(compare_num)]
        f_60 = [[] for i in range(compare_num)]
        F = [[] for i in range(compare_num)]
        # print(F)
        CD_1 = [[] for i in range(compare_num)]
        CD_2 = [[] for i in range(compare_num)]
        NC = [[] for i in range(compare_num)]
        object_id = [[] for i in range(compare_num)]


        # missing_part = check_missing_part(exp_name_list[0], shape_list, epoch)
        # print(missing_part)
        for method_idx, method_item in enumerate(methods_name):
            print(method_item)
            # if method_idx>0:
            #     continue

            save_dir = "abc_%s_%s_%dp_epoch_final_compare_all"%(method_item, type_item, n_points)
            if not os.path.exists(save_dir):
                print(save_dir)
                os.mkdir(save_dir)

            with open(shape_name_dir, 'r') as fp:
                lines = fp.readlines()
            shape_items = [line.split('/')[1].strip() for line in lines if shape_list[0] == line.split('/')[0]]
            print(len(shape_items))
           
            
            for shape_item in shape_items:
                # if shape_item=="00993917_4049b13b8ff84e59b2cfc43a_trimesh_000":
                #     print()
                #     continue
                # print(shape_item)
                # if shape_item=="cylinder100k_ddist_minmax":
                #     continue
                shape_name = shape_item
                if type_item[:7] == "density":
                    original_shape_name = shape_name.split("_d")[0]
                elif type_item[:5]=="noise":
                    original_shape_name = shape_name.split("_n")[0]
                else:
                    original_shape_name = shape_name

                gt_shape_path = "%s/test/%s/%s/gt.ply" % (
                    base_dir,
                    shape_list[0],
                    shape_item,
                )

                npy_path = "%s/%s.npy"%(save_dir, original_shape_name)

                
                if False and os.path.exists(npy_path):

                    
                    data = np.load(npy_path, allow_pickle=True)


                    ######### fixbug!!!!!!!!!!!!
                    gt_pointcloud = data.item()["tp"]
                    gt_normals = data.item()["tn"]
                    scale = data.item()["scale"]
                    center = data.item()["center"]
                        
                    f_10[method_idx].append(data.item()["f-score"])
                    f_15[method_idx].append(data.item()["f-score-15"])
                    f_20[method_idx].append(data.item()["f-score-20"])
                    f_30[method_idx].append(data.item()["f-score-30"])
                    f_40[method_idx].append(data.item()["f-score-40"])
                    f_50[method_idx].append(data.item()["f-score-50"])
                    f_60[method_idx].append(data.item()["f-score-60"])
                    NC[method_idx].append(data.item()["normals"])
                    CD_1[method_idx].append(data.item()["chamfer-L1"])
                    CD_2[method_idx].append(data.item()["chamfer-L2"])
                    # print(shape_item, data.item()["f-score"])
                    # print(shape_item, eval_dict_mesh["f-score"])
                else:
                    # original_name = shape_item.split("_dd")[0]
                    
                    if gt_shape_path in gt_cache.keys():
                        gt_pointcloud, gt_normals, center, scale = gt_cache[gt_shape_path]
                        print("use gt")
                    else:
                        # file_dir = "abc_%s_%s_%dp_epoch_final_compare_all/%s.npy"%("Our", type_item, n_points, shape_item)
                        # data = np.load(file_dir, allow_pickle=True)
                        # # print(method_name[idx])
                        # gt_pointcloud = data.item()["tp"]
                        # gt_normals = data.item()["tn"]

                        gt_mesh = trimesh.load_mesh(gt_shape_path)
                        # gt_normal_path = "/apdcephfs/share_1467498/home/runsongzhu/IGR/data/raw_data/optim_data/pclouds/%s.normals" % (
                        #     original_name,
                        # )


                        if isinstance(gt_mesh, trimesh.Trimesh):
                            gt_pointcloud, idx_pts = gt_mesh.sample(n_points, return_index=True)
                            gt_pointcloud = gt_pointcloud.astype(np.float32)
                            gt_normals = gt_mesh.face_normals[idx_pts]
                        elif isinstance(gt_mesh, trimesh.PointCloud):
                            print(";;;;")
                            # gt_pointcloud = gt_mesh.vertices.astype(np.float32)
                            # gt_normals = None
                            plydata = PlyData.read(gt_shape_path)
                            gt_pointcloud = np.stack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]], axis=1)
                            gt_normals = None #np.stack([plydata["vertex"]["nx"], plydata["vertex"]["ny"], plydata["vertex"]["nz"]], axis=1)
                        else:
                            raise RuntimeError("Unknown data type!")



                        N = gt_pointcloud.shape[0]
                        center = gt_pointcloud.mean(0)
                        scale = np.abs(gt_pointcloud - center).max()
                        # print(center, scale)


                        gt_pointcloud -= center
                        gt_pointcloud /= scale
                        index = np.random.choice(gt_pointcloud.shape[0], size=num_pts, replace=False)
                        gt_pointcloud = gt_pointcloud[index]
                        # save_path = "/data3/brian22/occupancy_networks/temp/"
                        # export_pointcloud(gt_pointcloud, os.path.join(save_path, f'{shape_item}_gt.ply'), False)
                        gt_cache.update({gt_shape_path: (gt_pointcloud, gt_normals, center, scale)})
                        


                    target_pts = gt_pointcloud
                    target_normals = gt_normals

                    for sample_idx in range(10):
                        if True:


                            # exp_base_dir = "/apdcephfs/share_1467498/datasets/runsongzhu/IGR/%s/exps/" % exp_name_list[method_idx]
                            # timestamps = os.listdir(os.path.join(exp_base_dir, shape_item))
                            # timestamp = sorted(timestamps)[-1]

                            file_data = f"%s/%s/%s.ply" % (
                                base_dir2,
                                shape_item,
                                f'fake-z{sample_idx}',
                            )

                            if not os.path.exists(file_data):
                                print('File does not exist:', file_data)
                                continue
                            mesh = trimesh.load(
                                file_data, process=False
                            )
                            print("check this!!!!", np.linalg.norm(mesh.vertices, ord=2, axis=1).max())


                        #mesh.vertices -= center
                        #mesh.vertices /= scale
                        center = mesh.vertices.mean(0)
                        mesh.vertices -= center
                        scale = np.abs(mesh.vertices).max()
                        mesh.vertices /= scale
                        # save_path = "/data3/brian22/occupancy_networks/temp/"
                        # export_pointcloud(mesh.vertices, os.path.join(save_path, f'{shape_item}_partial_{sample_idx}.ply'), False)
                        # scale = 1.
                        # print(np.abs(gt_pointcloud).max())
                        thresholds = np.linspace(1.0 / 1000, 1, 1000)
                        eval_dict_mesh = evaluator.eval_mesh(shape_item, mesh, target_pts, target_normals,save_dir, thresholds=thresholds,center=center, scale=center)
                        # print(eval_dict_mesh)

                        fpd[method_idx].append(eval_dict_mesh['fpd-score'])
                        f_10[method_idx].append(eval_dict_mesh["f-score"])
                        f_15[method_idx].append(eval_dict_mesh["f-score-15"])
                        f_20[method_idx].append(eval_dict_mesh["f-score-20"])
                        f_30[method_idx].append(eval_dict_mesh["f-score-30"])
                        f_40[method_idx].append(eval_dict_mesh["f-score-40"])
                        f_50[method_idx].append(eval_dict_mesh["f-score-50"])
                        f_60[method_idx].append(eval_dict_mesh["f-score-60"])
                        NC[method_idx].append(eval_dict_mesh["normals"])
                        CD_1[method_idx].append(eval_dict_mesh["chamfer-L1"])
                        CD_2[method_idx].append(eval_dict_mesh["chamfer-L2"])
                    
                
                # acc_metric[idx].append(eval_dict_mesh["accuracy"])
                # complete_metric[idx].append(eval_dict_mesh["completeness"])

    save_data = {}
    save_data["methods"] = methods_name
    save_data["score_10"] = [ round(np.array(f_10_item).mean(),3) for f_10_item in f_10] #np.mean(np.array(f_10))
    save_data["score_15"] = [ round(np.array(f_15_item).mean(),3) for f_15_item in f_15] #np.mean(np.array(f_15[idx]))
    save_data["score_20"] = [ round(np.array(f_20_item).mean(),3) for f_20_item in f_20] #np.mean(np.array(f_20))
    save_data["score_30"] = [round(np.array(f_30_item).mean(), 3) for f_30_item in f_30]  # np.mean(np.array(f_10))
    save_data["score_40"] = [round(np.array(f_40_item).mean(), 3) for f_40_item in f_40]  # np.mean(np.array(f_15[idx]))
    save_data["score_50"] = [round(np.array(f_50_item).mean(), 3) for f_50_item in f_50]  # np.mean(np.array(f_20))
    save_data["score_60"] = [round(np.array(f_60_item).mean(), 3) for f_60_item in f_60]  # np.mean(np.array(f_20))
    #save_data["CD1"] = [ round(np.array(CD_1_item).mean() * 100.0,3) for CD_1_item in CD_1] #np.mean(np.array(CD_1))
    save_data["CD1"] = [round(np.array(CD_1_item).reshape(-1, 10).min(-1).mean() * 100.0, 3) for CD_1_item in CD_1]
    save_data['fpd'] = [round(np.array(fpd_item).reshape(-1, 10).min(-1).mean(), 3) for fpd_item in CD_1]
    # save_data["CD2"] = [ np.array(CD_2_item).mean() for CD_2_item in CD_2]# np.mean(np.array(CD_2))
    save_data["NC"] = [ round(np.array(NC_item).mean(),3) for NC_item in NC] # np.mean(np.array(NC))
    # top_list = [ list(np.argsort(np.array(CD_2_item)- np.array(CD_2[0]) )[:20]) for CD_2_item in CD_2] 
    # # print(top_list)
    # # print(object_id)
    # top_list = [[object_id[idx][name_idx] for name_idx in top_list_item] for idx,top_list_item in enumerate(top_list)]
    # print(top_list)

    # top_list = [ list(np.argsort(np.array(CD_2_item))[-20:]) for CD_2_item in CD_2] 
    # # print(top_list)
    # # print(object_id)
    # top_list = [[object_id[idx][name_idx] for name_idx in top_list_item] for idx,top_list_item in enumerate(top_list)]
    # print(top_list[0])

    df = pd.DataFrame(save_data)
    df.to_csv("%s/%s_%s_%s_thingi_compare_compare_traditional_sampling.csv" % (metric_save_dir, methods_name[0], dataset, shape_list[0]))
    # print("acc_metric", np.mean(np.array(acc_metric[idx])))
    # print("complete_metric", np.mean(np.array(complete_metric[idx])))

    # Creating dataset
    # np.random.seed(10)

    