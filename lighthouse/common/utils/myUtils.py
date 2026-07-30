import copy
import json
import os
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import yaml
from easydict import EasyDict


# 两种事件窗口格式切换
# cxw_span：(center,length)
# xx_span： (start,end)
def span_cxw_to_xx(cxw_spans):
    """
    args:
        cxw_spans: 
            tensor, 最后一维为2, 对应窗口格式(center,length)
    output:
            tensor, 形状于输入一致, 最后一维维度为2, 对应窗口格式(start,end)
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)

def span_xx_to_cxw(xx_spans):
    """
        span_cxw_to_xx的逆操作
    """
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)

# 几种常用的IoU

# IoU
def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union

# GIoU 广义事件交并比 不仅考虑了交并比（IoU），还考虑了两个窗口的包围区域。
def generalized_temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)
    # 利用none插入一个维度，借助广播机制完成窗口之间的两两比较
    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, , 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]

        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # pred_spans = pred_spans.repeat(1, n_spans, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, max_v_l, 2)
            # tgt_spans = tgt_spans.view(1, n_spans, 2).repeat(bs * num_queries, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, 2)
            # cost_span = pred_spans[tgt_spans]
            # cost_span = cost_span.view(bs * num_queries, n_spans)

            # giou
            cost_giou = 0

        # Final cost matrix
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )

# basic_utils
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))

def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

# 填充特征，使长度统一
def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths
    
# model_utils
def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable

# EMA, 指数移动平均。一种用于平滑时间序列数据的技术。通过对数据进行加权平均来减少噪音和波动，从而提取出数据的趋势。# decay是EMA的衰减因子
class ModelEMA(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e,
                     m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


# loss_func
class CTC_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CTC_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, vid_feat, txt_feat, pos_mask, src_vid_mask=None, src_txt_mask=None):
        # vid_feat: (bs, t, d)
        # txt_feat: (bs, n, d)
        # pos_mask: (bs, t)
        # src_vid_mask: (bs, t) or None
        # src_txt_mask: (bs, n) or None
        bs = vid_feat.size(0)
        t = vid_feat.size(1)
        n = txt_feat.size(1)
        d = vid_feat.size(2)
        # normalize the feature vectors
        vid_feat = F.normalize(vid_feat, dim=2) # (bs, t, d)
        txt_feat = F.normalize(txt_feat, dim=2) # (bs, n, d)
        # compute the global text feature by mean pooling
        if src_txt_mask is not None:
            src_txt_mask = src_txt_mask.unsqueeze(-1) # (bs, n, 1)
            txt_feat = txt_feat * src_txt_mask # (bs, n, d)
            txt_global = torch.sum(txt_feat, dim=1) / torch.sum(src_txt_mask, dim=1) # (bs, d)
        else:
            txt_global = torch.mean(txt_feat, dim=1) # (bs, d)
        # compute the similarity matrix
        sim_mat = torch.bmm(vid_feat, txt_global.unsqueeze(-1)).squeeze(-1) # (bs, t)
        # apply the video mask if given
        if src_vid_mask is not None:
            sim_mat = sim_mat * src_vid_mask # (bs, t)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, t)
        labels = pos_mask.long() # (bs, t)
        # compute the binary cross entropy loss with logits
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()) # scalar
        # return the loss
        return loss


class VTCLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(VTCLoss, self).__init__()
        self.temperature = temperature

    def forward(self, src_txt, src_vid):
        # src_txt: (bs, h_dim)
        # src_vid: (bs, h_dim)
        bs = src_txt.size(0)
        h_dim = src_txt.size(1)
        # normalize the feature vectors
        src_txt = F.normalize(src_txt, dim=1)
        src_vid = F.normalize(src_vid, dim=1)
        # compute the similarity matrix
        sim_mat = torch.mm(src_txt, src_vid.t()) # (bs, bs)
        # create the positive and negative masks
        pos_mask = torch.eye(bs).bool().to(sim_mat.device) # (bs, bs)
        neg_mask = ~pos_mask # (bs, bs)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, bs)
        labels = torch.arange(bs).to(sim_mat.device) # (bs,)
        # compute the cross entropy loss for text-to-video and video-to-text
        loss_t2v = F.cross_entropy(logits, labels) # scalar
        loss_v2t = F.cross_entropy(logits.t(), labels) # scalar
        # return the average loss
        return (loss_t2v + loss_v2t) / 2

class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

def write_log(opt, epoch_i, loss_meters, metrics=None, mode='train'):
    # log
    if mode == 'train':
        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i+1,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        filename = opt.train_log_filepath
    else:
        to_write = opt.eval_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]),
            eval_metrics_str=json.dumps(metrics))
        filename = opt.eval_log_filepath
    
    with open(filename, "a") as f:
        f.write(to_write)


def save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch_i,
        "opt": opt
    }
    torch.save(checkpoint, opt.ckpt_filepath)

def rename_latest_to_best(latest_file_paths):
    best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
    for src, tgt in zip(latest_file_paths, best_file_paths):
        os.renames(src, tgt)

# config.py

class BaseOptions(object):
    def __init__(self, model, dataset, feature, resume, domain):
        self.model = model
        self.dataset = dataset
        self.feature = feature
        self.resume = resume
        self.domain = domain
        self.opt = {}

    @property
    def option(self):
        if len(self.opt) == 0:
            raise RuntimeError('option is empty. Did you run parse()?')
        return self.opt

    def update(self, yaml_file):
        with open(yaml_file, 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            self.opt.update(yml)

    def parse(self):
        base_cfg = 'configs/base.yml'
        feature_cfg = f'configs/feature/{self.feature}.yml'
        model_cfg = f'configs/model/{self.model}.yml'
        dataset_cfg = f'configs/dataset/{self.dataset}.yml'
        cfgs = [base_cfg, feature_cfg, model_cfg, dataset_cfg]
        for cfg in cfgs:
            self.update(cfg)

        self.opt = EasyDict(self.opt)

        if self.resume:
            self.opt.results_dir = os.path.join(self.opt.results_dir, self.model, f"{self.dataset}_finetune", self.feature)
        else:
            self.opt.results_dir = os.path.join(self.opt.results_dir, self.model, self.dataset, self.feature)
            if self.domain:
                self.opt.results_dir = os.path.join(self.opt.results_dir, self.domain)

        self.opt.ckpt_filepath = os.path.join(self.opt.results_dir, self.opt.ckpt_filename)
        self.opt.train_log_filepath = os.path.join(self.opt.results_dir, self.opt.train_log_filename)
        self.opt.eval_log_filepath = os.path.join(self.opt.results_dir, self.opt.eval_log_filename)

        # feature directory
        v_feat_dirs = None
        t_feat_dir = None
        a_feat_dirs = None
        a_feat_types = None
        t_feat_dir_pretrain_eval = None

        if self.dataset == 'qvhighlight_pretrain':
            
            dataset = self.dataset.replace('_pretrain', '')

            if self.feature == 'clip_slowfast_pann':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{dataset}/clip', f'features/{dataset}/slowfast']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{dataset}/clip_text_subs_train'
                t_feat_dir_pretrain_eval = f'/media/sda/szr/lighthouse-main/features/{dataset}/clip_text'
                a_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{dataset}/pann']
                a_feat_types = self.opt.a_feat_types
                
            elif self.feature == 'clip_slowfast':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{dataset}/clip', f'/media/sda/szr/lighthouse-main/features/{dataset}/slowfast']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{dataset}/clip_text_subs_train'
                t_feat_dir_pretrain_eval = f'/media/sda/szr/lighthouse-main/features/{dataset}/clip_text'

            elif self.feature == 'clip':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{dataset}/clip']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{dataset}/clip_text_subs_train'
                t_feat_dir_pretrain_eval = f'/media/sda/szr/lighthouse-main/features/{dataset}/clip_text'

            else:
                raise ValueError(f'For pre-train, features should include CLIP, but {self.feature} is used.')
        
        else:
            if self.feature == 'clip_slowfast_pann':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip', f'/media/sda/szr/lighthouse-main/features/{self.dataset}/slowfast']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip_text'
                a_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/pann']
                a_feat_types = self.opt.a_feat_types
                
            elif self.feature == 'clip_slowfast':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip', f'/media/sda/szr/lighthouse-main/features/{self.dataset}/slowfast']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip_text'

            elif self.feature == 'clip':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip_text'

            elif self.feature == 'resnet_glove':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/resnet']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{self.dataset}/glove'

            elif self.feature == 'i3d_clip':
                v_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/i3d']
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clip_text'

            elif self.feature == 'clap':
                a_feat_dirs = [f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clap']
                a_feat_types = self.opt.a_feat_types
                t_feat_dir = f'/media/sda/szr/lighthouse-main/features/{self.dataset}/clap_text'

        self.opt.v_feat_dirs = v_feat_dirs
        self.opt.t_feat_dir = t_feat_dir
        self.opt.a_feat_dirs = a_feat_dirs
        self.opt.a_feat_types = a_feat_types
        self.opt.t_feat_dir_pretrain_eval = t_feat_dir_pretrain_eval

    def clean_and_makedirs(self):
        if 'results_dir' not in self.opt:
            raise RuntimeError('results_dir is not set in self.opt. Did you run parse()?')
        
        if os.path.exists(self.opt.results_dir):
            shutil.rmtree(self.opt.results_dir)

        os.makedirs(self.opt.results_dir, exist_ok=True)