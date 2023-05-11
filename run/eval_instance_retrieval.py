import os
import random
import numpy as np
import logging
import argparse
import urllib

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo

from MinkowskiEngine import SparseTensor
from util import config
from util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from run.distill import get_model

from dataset.label_constants import *


def get_parser():
    """Parse the config file."""

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                        default='config/scannet/eval_openseg.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/test_ours_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    """Define logger."""

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')


def precompute_text_related_properties(labelset_name, args):
    """pre-compute text features, labelset, palette, and mapper."""

    if labelset_name == "scannet20":
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other'  # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'scannet200':
        labelset = list(SCANNET_LABELS_200)
        palette = get_palette(colormap='matterport_160')
    elif labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    else:
        raise NotImplementedError(f"Unknown labelset: {labelset_name}")

    text_features = extract_text_feature(labelset, args)
    labelset.append('unlabeled')
    return text_features, labelset, palette


def main():
    """Main function."""

    args = get_parser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False
        print("Single GPU mode")

    model = get_model(args).cuda()
    logger = get_logger()
    logger.info(args)

    if args.feature_type == 'fusion':
        pass  # do not need to load weight
    elif is_url(args.model_path):  # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
            logger.info('Loaded a parallel model')

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default, we do not use the point color as input
        args.input_color = False

    from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                  datapath_prefix_feat=args.data_root_2d_fused_feature,
                                  voxel_size=args.voxel_size,
                                  split=args.split, aug=False,
                                  memcache_init=False, eval_all=True, identifier=6797,
                                  input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                             drop_last=False, collate_fn=collation_fn_eval_all,
                                             sampler=val_sampler)

    # ####################### Test ####################### #
    torch.backends.cudnn.enabled = False
    print(args.save_folder)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
    exit(0)

    if args.save_feature_as_numpy:  # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    feature_type = args.feature_type
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion.
        # Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True

    labelset_name = "scannet200"
    text_features, labelset, palette = precompute_text_related_properties(labelset_name, args)

    with torch.no_grad():
        model.eval()

        val_loader.dataset.offset = 0
        logger.info("\nEvaluation {} out of {} runs...\n".format(1, args.test_repeats))

        for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_loader)):
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            coords = coords[inds_reverse, :]
            pcl = coords[:, 1:].cpu().numpy()

            if feature_type == 'distill':
                predictions = model(sinput)
                predictions = predictions[inds_reverse, :]
                pred = predictions.half() @ text_features.t()
                logits_pred = torch.max(pred, 1)[1].cpu()
            elif feature_type == 'fusion':
                predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                pred = predictions.half() @ text_features.t()
                logits_pred = torch.max(pred, 1)[1].detach().cpu()
                if mark_no_feature_to_unknown:
                    # some points do not have 2D features from 2D feature fusion.
                    # Directly assign 'unknown' label to those points during inference.
                    logits_pred[~mask[inds_reverse]] = len(labelset) - 1
            elif feature_type == 'ensemble':
                feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                # pred_fusion = feat_fuse.half() @ text_features.t()
                pred_fusion = (feat_fuse / (feat_fuse.norm(dim=-1, keepdim=True) + 1e-5)).half() @ text_features.t()

                predictions = model(sinput)
                predictions = predictions[inds_reverse, :]
                # pred_distill = predictions.half() @ text_features.t()
                pred_distill = (predictions / (
                            predictions.norm(dim=-1, keepdim=True) + 1e-5)).half() @ text_features.t()

                feat_ensemble = predictions.clone().half()
                mask_ = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
                feat_ensemble[mask_] = feat_fuse[mask_]
                pred = feat_ensemble @ text_features.t()
                logits_pred = torch.max(pred, 1)[1].detach().cpu()

                predictions = feat_ensemble  # if we need to save the features
            else:
                raise NotImplementedError

            if args.save_feature_as_numpy:
                scene_name = val_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                np.save(
                    os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)),
                    predictions.cpu().numpy())


if __name__ == '__main__':
    main()
