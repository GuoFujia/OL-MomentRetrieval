import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lighthouse.common.matcher import build_matcher
from lighthouse.common.misc import accuracy
from lighthouse.common.moment_transformer import build_transformer
from lighthouse.common.position_encoding import build_position_encoding
from lighthouse.common.utils.span_utils import temporal_iou, generalized_temporal_iou, span_cxw_to_xx
from lighthouse.common.compressor import SimpleCompressor
from lighthouse.common.compressor import ResidualCompressor, CrossAttentionResidualCompressor, CrossAttentionResidualCompressorWithQuery
from lighthouse.common.compressor import CompressorWithExternalWeights, CompressorWithWeightedKV, CompressorWithPostWeighting, CompressorWithSmoothMechanism, CompressorWithPositionWeights
from lighthouse.common.compressor import ConcatCompressor, CrossAttentionConcatCompressor
from lighthouse.common.compressor import MultimodalTokenCompressor, TextGuidedCompressor, TextPresentGuidedCompressor
from lighthouse.common.memoryModule import InterMemory, InterMemorySeq

import time
import functools
from typing import Any, Callable


def countTime(func: Callable) -> Callable:
    """装饰器：记录函数执行的开始时间、结束时间和耗时，写入日志文件。

    输入:
        func: 被装饰的可调用对象
    输出:
        wrapper: 包装后的函数，执行时自动记录时间信息到 ./time_log.txt
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # 获取类名（如果是类方法）
        class_name = ""
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__ + "."
        func_name = f"{class_name}{func.__name__}"

        log_path = "./time_log.txt"

        start_time = time.perf_counter()
        with open(log_path, "a") as f:
            f.write(f"[Start] {func_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start_time
        with open(log_path, "a") as f:
            f.write(
                f"[End] {func_name} -> "
                f"耗时: {elapsed:.6f}s, "
                f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
        return result
    return wrapper


class OLMomentDETR(nn.Module):
    """在线版本的 Moment-DETR 模型，用于视频时刻定位。

    在离线版本的基础上增加了长期记忆压缩和跨片段记忆模块，
    支持在线逐 chunk 推理场景。
    """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, weight_alpha, attn_weight, future_memory_sample_len,
                 aux_loss=False, max_v_l=75, span_loss_type="l1",
                 use_txt_pos=False, n_input_proj=2, aud_dim=0, compress_len=10,
                 use_vid_compression=True, use_inter_memory=True, memory_topk=4,
                 memory_slots=64,
                 compressor_output_mode="dual_intra",
                 inter_memory_output_mode="future_only",
                 ablate_delta_stream=False,
                 use_saliency_bias=True,
                 use_distance_bias=True,
                 slot_select_mode="header",
                 disable_phase_alignment=False,
                 fixed_phase_start=0,
                 inter_memory_use_diversity_loss=True,
                 inter_memory_use_struct_loss=True,
                 inter_memory_diversity_loss_coef=0.001,
                 inter_memory_struct_loss_coef=0.01,
                 inter_memory_slot_len=None,
                 inter_memory_legacy_v1=False):
        """初始化模型。

        参数:
            transformer: transformer 架构模块
            position_embed: 视频位置编码模块
            txt_position_embed: 文本位置编码模块
            txt_dim: int, 文本特征输入维度
            vid_dim: int, 视频特征输入维度
            num_queries: int, 查询数量（检测槽位数），决定模型能检测的最大时刻数
            input_dropout: float, 输入 dropout 率
            weight_alpha: float, 压缩器的权重 alpha 参数
            attn_weight: float, 压缩器的注意力权重参数
            future_memory_sample_len: int, 未来记忆采样长度
            aux_loss: bool, 是否使用辅助解码损失
            max_v_l: int, 视频最大 clip 数
            span_loss_type: str, "l1" 为 (center-x, width) 回归，"ce" 为分类
            use_txt_pos: bool, 是否使用文本位置编码
            n_input_proj: int, 输入投影层数
            aud_dim: int, 音频特征维度
            compress_len: int, 压缩后的长期记忆长度
            use_vid_compression: bool, 是否使用视频压缩
            use_inter_memory: bool, 是否使用跨片段记忆
            memory_topk: int, 记忆模块的 topk 参数
            memory_slots: int, inter-memory 槽位数量 N_slot
            ablate_delta_stream: bool, 是否用 static 替代 delta 分支（E1）。
            use_saliency_bias: bool, 是否启用显著性 bias（E2）。
            use_distance_bias: bool, 是否启用距离 bias（E3）。
            slot_select_mode: str, slot 选择方式，'header' 或 'slot_mean_text'（E4）。
            disable_phase_alignment: bool, 是否关闭二次定位（E5）。
            fixed_phase_start: int, 关闭二次定位时的固定起点。
            inter_memory_use_diversity_loss: bool, 是否启用 inter-memory diversity loss。
            inter_memory_use_struct_loss: bool, 是否启用 inter-memory struct loss。
            inter_memory_diversity_loss_coef: float, diversity loss 系数。
            inter_memory_struct_loss_coef: float, struct loss 系数。
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2

        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.frame_class_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 输出维度为4，表示4个二分类的 logits
        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.compress_len = compress_len
        self.future_memory_sample_len = future_memory_sample_len
        self.use_inter_memory = use_inter_memory
        self.memory_topk = memory_topk
        self.memory_slots = memory_slots
        self.compressor_output_mode = compressor_output_mode
        self.inter_memory_output_mode = inter_memory_output_mode
        self.inter_memory_use_diversity_loss = inter_memory_use_diversity_loss
        self.inter_memory_use_struct_loss = inter_memory_use_struct_loss
        self.inter_memory_diversity_loss_coef = inter_memory_diversity_loss_coef
        self.inter_memory_struct_loss_coef = inter_memory_struct_loss_coef
        self.inter_memory_legacy_v1 = inter_memory_legacy_v1
        self.inter_memory_slot_len = inter_memory_slot_len

        # saliency_score_dict: 记录每个 (qid, vid) 对对应的显著性分数历史
        # 结构: {(qid, vid): {'start': int, 'weight': np.ndarray}}
        #   - start: 当前 vid 对于 qid 的采样起始帧索引
        #   - weight: 二维 float 数组, weight[i][j] 代表从 start 开始的第 i 帧的第 j 次预测 saliency_score
        self.saliency_score_dict = {}

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj

        self.use_vid_compression = use_vid_compression
        if self.use_vid_compression:
            self.memory_compressor = CrossAttentionResidualCompressorWithQuery(
                dimension=hidden_dim, compress_len=self.compress_len,
                weight_alpha=weight_alpha, attn_weight=attn_weight,
                compressor_output_mode=self.compressor_output_mode,
                ablate_delta_stream=ablate_delta_stream,
                use_saliency_bias=use_saliency_bias,
                use_distance_bias=use_distance_bias,
            )

        if self.use_inter_memory and self.future_memory_sample_len > 0:
            if self.inter_memory_slot_len is not None:
                default_slot_len = self.inter_memory_slot_len
            else:
                default_slot_len = self.compress_len + future_memory_sample_len * 2
            self.inter_memory = InterMemorySeq(
                d_model=hidden_dim, future_length=future_memory_sample_len,
                memory_slots=self.memory_slots,
                slot_len=default_slot_len, compress_len=self.compress_len,
                similarity_threshold=0.9, topk=self.memory_topk,
                diversity_loss_weight=0.5, optimization_interval=1, optimizer_lr=1e-4,
                slot_select_mode=slot_select_mode,
                disable_phase_alignment=disable_phase_alignment,
                fixed_phase_start=fixed_phase_start,
                legacy_v1=inter_memory_legacy_v1,
            )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.aux_loss = aux_loss

    @countTime
    def forward(self, src_txt, src_txt_mask, src_vid, memory_len, src_vid_mask, chunk_idx,
                qid_vid, short_memory_start, src_aud=None, src_aud_mask=None,
                long_memory_weight=None, gt_frames_in_long_memory=None,
                frame_index_long=None, frame_index_long_mask=None, center_index=None,
                delta_gt=None, delta_gt_valid=None):
        """在线模型的前向传播。

        输入:
            src_txt: [batch_size, L_txt, D_txt] 文本特征
            src_txt_mask: [batch_size, L_txt] 文本掩码
            src_vid: [batch_size, L_vid, D_vid] 视频特征，
                L_vid = long_memory_sample_length + short_memory_sample_length + future_memory_sample_length
            src_vid_mask: [batch_size, L_vid] 视频掩码
            memory_len: list, 三种记忆的采样长度
            chunk_idx: [batch_size] 当前 chunk 索引
            qid_vid: [batch_size] 样本标识，用于匹配 saliency_score_dict
            short_memory_start: [batch_size] 短期记忆起始帧索引
            src_aud: 可选，音频特征
            src_aud_mask: 可选，音频掩码
            long_memory_weight: [batch_size, L_vid] 训练时使用的外部权重
            gt_frames_in_long_memory: [batch_size] 历史帧与 gt 的相交帧数量

        输出: dict 包含:
            pred_spans: 归一化的时刻坐标 (center_x, width)
            pred_logits: 所有查询的类别预测
            frame_pred: short 记忆的逐帧预测概率
            saliency_scores: short 记忆的逐帧显著性分数
            chunk_idx: chunk 索引
        """
        # 拼接音频特征
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        # 投影输入特征
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        # 分离长期记忆和当前帧
        long_vid = src_vid[:, :memory_len[0]]
        long_vid_mask = src_vid_mask[:, :memory_len[0]]
        present_vid = src_vid[:, memory_len[0]:]

        # 压缩长期记忆
        memory_struct_loss = None
        if self.use_vid_compression:
            # 训练时使用数据集提供的显著性分数作为 extern_weight
            if self.training:
                extern_weight = long_memory_weight
            else:
                # 推理时使用过去推理的结果构建 extern_weight
                all_weights = []
                extern_weight = None

                for batch_idx in range(len(qid_vid)):
                    output = self.get_long_memory_weight_by_qid_vid(
                        qid_vid[batch_idx][0], qid_vid[batch_idx][1]
                    )

                    if output is None:
                        all_weights = []
                        extern_weight = None
                        break

                    indexed_weight = self.index_weight(
                        output[1], output[0], short_memory_start[batch_idx], memory_len[0]
                    )
                    all_weights.append(torch.from_numpy(indexed_weight).float())

                # 只有在所有 batch 都成功获取权重时才 stack
                if len(all_weights) == len(qid_vid):
                    extern_weight = torch.stack(all_weights, dim=0)
                else:
                    extern_weight = None

            # 确保 extern_weight 与 long_vid 在相同设备上
            if extern_weight is not None:
                extern_weight = extern_weight.to(long_vid.device)

            if torch.isnan(long_vid).any():
                print("WARNING: long_vid contains NaN!")

            # 执行压缩
            compress_long_res, compress_long_mask = self.memory_compressor(
                Fv=long_vid, vid_mask=long_vid_mask,
                Ft=src_txt, text_mask=src_txt_mask,
                extern_weight=extern_weight,
                prev_saliency=extern_weight,
                frame_index=frame_index_long,
                center_index=center_index,
            )
            history_static = None
            history_delta = None
            if isinstance(compress_long_res, dict):
                history_static = compress_long_res.get("static")
                history_delta = compress_long_res.get("delta")
                compress_long_vid = compress_long_res.get("fused")
            else:
                compress_long_vid = compress_long_res

            if torch.isnan(compress_long_vid).any():
                print("WARNING: compress_long_vid contains NaN!")
                print("long_vid:", long_vid)
                print("src_txt:", src_txt)
                print("src_txt_mask:", src_txt_mask)
                print("long_vid_mask:", long_vid_mask)
                print("extern_weight:", extern_weight)

            # 将压缩后的长期记忆和短期/future 记忆拼接得到新的 src_vid, src_vid_mask
            src_vid = torch.cat([compress_long_vid, src_vid[:, memory_len[0]:]], dim=1)
            src_vid_mask = torch.cat([compress_long_mask, src_vid_mask[:, memory_len[0]:]], dim=1)
            memory_len[0] = self.compress_len

        # 使用压缩后的历史记忆维护 interMemory
        if self.use_inter_memory and self.inter_memory is not None and self.use_vid_compression:
            # 将 delta_gt 对齐到 inter-memory 当前特征空间和时间长度
            delta_gt_for_memory = None
            delta_gt_valid_for_memory = None
            if delta_gt is not None:
                delta_gt_for_memory = delta_gt.to(src_vid.device)
                if delta_gt_for_memory.shape[-1] != present_vid.shape[-1]:
                    # 数据集侧 delta_gt 基于原始短期特征构造，这里投影到 hidden_dim
                    delta_gt_for_memory = self.input_vid_proj(delta_gt_for_memory)
                target_len = present_vid.shape[1]
                current_len = delta_gt_for_memory.shape[1]
                if current_len < target_len:
                    pad_len = target_len - current_len
                    pad_feat = torch.zeros(
                        delta_gt_for_memory.shape[0],
                        pad_len,
                        delta_gt_for_memory.shape[-1],
                        device=delta_gt_for_memory.device,
                        dtype=delta_gt_for_memory.dtype,
                    )
                    delta_gt_for_memory = torch.cat([delta_gt_for_memory, pad_feat], dim=1)
                elif current_len > target_len:
                    delta_gt_for_memory = delta_gt_for_memory[:, :target_len]
                if delta_gt_valid is None:
                    delta_gt_valid_for_memory = torch.ones(
                        delta_gt_for_memory.shape[0],
                        delta_gt_for_memory.shape[1],
                        device=delta_gt_for_memory.device,
                        dtype=torch.bool,
                    )
                else:
                    delta_gt_valid_for_memory = delta_gt_valid.to(src_vid.device).bool()
                    valid_len = delta_gt_valid_for_memory.shape[1]
                    if valid_len < target_len:
                        pad_len = target_len - valid_len
                        pad_valid = torch.zeros(
                            delta_gt_valid_for_memory.shape[0],
                            pad_len,
                            device=delta_gt_valid_for_memory.device,
                            dtype=torch.bool,
                        )
                        delta_gt_valid_for_memory = torch.cat([delta_gt_valid_for_memory, pad_valid], dim=1)
                    elif valid_len > target_len:
                        delta_gt_valid_for_memory = delta_gt_valid_for_memory[:, :target_len]

            # 确保 InterMemory 的参数与输入同设备
            if self.inter_memory.memory.device != src_vid.device:
                self.inter_memory.memory.data = self.inter_memory.memory.data.to(src_vid.device)
                self.inter_memory.future_embedding.data = self.inter_memory.future_embedding.data.to(src_vid.device)
                if not self.inter_memory_legacy_v1:
                    self.inter_memory.header_tokens.data = self.inter_memory.header_tokens.data.to(src_vid.device)

            future_vid, future_mask = self.inter_memory.forward_with_controls(
                history_features=compress_long_vid, mask=compress_long_mask,
                current_features=present_vid,
                history_static=history_static,
                history_delta=history_delta,
                text_query=src_txt,
                text_query_mask=src_txt_mask,
                delta_gt=delta_gt_for_memory,
                delta_gt_valid=delta_gt_valid_for_memory,
                output_mode=self.inter_memory_output_mode,
                allow_infer_memory_update=False,
            )
            memory_struct_loss = self.inter_memory.last_memory_struct_loss

            # 将未来部分拼接
            if future_vid.shape[1] > self.future_memory_sample_len:
                # legacy_full 兼容：只拼接末尾 future 段，避免重复拼接历史/当前段
                future_vid = future_vid[:, -self.future_memory_sample_len:]
                future_mask = future_mask[:, -self.future_memory_sample_len:]
            src_vid = torch.cat([src_vid, future_vid], dim=1)
            src_vid_mask = torch.cat([src_vid_mask, future_mask], dim=1)

        # 拼接视频和文本的特征与掩码
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        # 生成位置编码
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        # 通过 transformer
        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        # hs: (num_decoder_layers, batch_size, num_queries, hidden_dim)
        # memory: (batch_size, L_vid+L_txt, hidden_dim)

        out = {}

        # span 预测和类别预测
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out['pred_logits'] = outputs_class[-1]
        out['pred_spans'] = outputs_coord[-1]
        if memory_struct_loss is not None:
            out["loss_memory_struct"] = memory_struct_loss

        # 帧预测结果和显著性分数（仅对 short memory 部分）
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        short_start = memory_len[0]
        short_end = short_start + memory_len[1]

        out['frame_pred'] = self.frame_class_embed(vid_mem)[:, short_start:short_end]  # (bsz, short_memory_length, 4)
        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)[:, short_start:short_end]  # (bsz, short_memory_length)

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        out["chunk_idx"] = chunk_idx  # [batch_size]

        # 维护和更新 saliency_score_dict（仅在推理时且使用压缩时）
        if self.use_vid_compression and not self.training and qid_vid is not None and short_memory_start is not None:
            self._update_saliency_score_dict(
                qid_vid, short_memory_start, out["saliency_scores"], memory_len
            )

        return out

    def _update_saliency_score_dict(self, qid_vid, short_memory_start, saliency_scores, memory_len):
        """更新 saliency_score_dict，记录每个 (qid, vid) 对的显著性分数历史。

        输入:
            qid_vid: list of (qid, vid) 样本标识列表
            short_memory_start: list 短期记忆起始索引
            saliency_scores: [batch_size, short_memory_len] 显著性分数
            memory_len: list 记忆长度列表
        """
        batch_size = saliency_scores.shape[0]
        weight_col = memory_len[1]

        for b_idx in range(batch_size):
            qid, vid = qid_vid[b_idx]
            current_short_start = short_memory_start[b_idx]
            current_saliency_score = saliency_scores[b_idx, :]
            current_length = current_saliency_score.size(0)
            current_saliency_score_np = current_saliency_score.cpu().detach().numpy()

            if (qid, vid) not in self.saliency_score_dict:
                # 初始化新条目
                weight = np.full((current_length, weight_col), np.nan)
                weight[:current_length, 0] = current_saliency_score_np
                self.saliency_score_dict[(qid, vid)] = {
                    'start': current_short_start,
                    'weight': weight
                }
            else:
                entry = self.saliency_score_dict[(qid, vid)]
                existing_start = entry['start']
                existing_weight = entry['weight']
                new_L = current_short_start - existing_start + current_length
                new_weight = np.full((new_L, weight_col), np.nan)
                new_weight[:existing_weight.shape[0], :weight_col] = existing_weight

                # 将当前预测结果填入对应位置的第一个空 NaN 槽
                for frame_idx, saliency_score in enumerate(current_saliency_score_np):
                    row_idx = current_short_start + frame_idx - existing_start
                    for col_idx, existing_val in enumerate(new_weight[row_idx]):
                        if np.isnan(existing_val):
                            if np.isnan(saliency_score):
                                print("WARNING: NaN detected in saliency score")
                            new_weight[row_idx, col_idx] = saliency_score
                            break

                entry['weight'] = new_weight

    def reset_saliency_score_dict(self):
        """重置显著性分数字典，清空所有历史记录。"""
        self.saliency_score_dict = {}

    def reset_inter_memory(self):
        """重新初始化 inter_memory，用于两个 epoch 之间重置状态。"""
        if self.use_inter_memory and self.inter_memory is not None:
            self.inter_memory.reset()

    def process_probs_by_avg(self, probs):
        """对概率值取平均。

        输入:
            probs: np.ndarray, 有效概率值数组
        输出:
            float, 平均概率值
        """
        return np.mean(probs, axis=0)

    def get_long_memory_weight_by_qid_vid(self, qid, vid):
        """通过 (qid, vid) 从 saliency_score_dict 获取长期记忆的权重。

        输入:
            qid: 查询 ID
            vid: 视频 ID
        输出:
            None: 如果没有对应记录
            (start, weight_array): start 为起始帧索引，weight_array 为归一化到 [0,1] 的权重数组
        """
        if (qid, vid) not in self.saliency_score_dict:
            return None

        entry = self.saliency_score_dict[(qid, vid)]
        dict_weight = entry["weight"]
        res = []

        for frame in dict_weight:
            # 筛除 NaN 后取平均
            valid_frame = frame[~np.isnan(frame)]

            if np.isnan(valid_frame).any():
                print("WARNING: all NaN in valid_frame")

            frame_weight = self.process_probs_by_avg(valid_frame)

            # 对 QV 数据集的分数进行缩放
            if ".0" in vid:
                frame_weight = frame_weight / 4

            # 归一化到 [0, 1]
            frame_weight = max(0.0, min(1.0, frame_weight))

            res.append(frame_weight)

        return entry["start"], np.array(res)

    def index_weight(self, weight_list, weight_start, short_start, return_len):
        """从权重列表中索引出 [weight_start, short_start) 之间的分数，并 pad 到 return_len 长度。

        输入:
            weight_list: np.ndarray, 权重数组
            weight_start: int, 权重起始索引
            short_start: int, 短期记忆起始索引
            return_len: int, 返回长度
        输出:
            np.ndarray, 长度为 return_len 的权重数组，有效值左侧对齐
        """
        weight_list = np.pad(weight_list, (weight_start, 0), 'constant', constant_values=0)

        if weight_start > short_start:
            return None

        raw_res = weight_list[:short_start]

        # 特殊情况：长视频的相邻样本放进一个 batch 时，部分历史帧分数可能尚未保存
        if len(weight_list) < short_start:
            raw_res = np.pad(weight_list, (0, short_start - len(weight_list)), 'constant', constant_values=0)

        if len(raw_res) < return_len:
            # 有效位在左侧，与 mask 保持一致
            raw_res = np.pad(raw_res, (0, return_len - len(raw_res)), 'constant', constant_values=0)
        else:
            raw_res = raw_res[-return_len:]

        return raw_res


class SetCriterionOl(nn.Module):
    """在线 Moment-DETR 的损失计算模块。

    主要计算以下损失:
        1) 帧级别的分类损失 [start, mid, end, irrelevant]
        2) saliency score 预测结果的损失
        3) span 级别的预测损失
    """

    def __init__(self, weight_dict, losses, saliency_margin=1, gamma=.0):
        """初始化损失计算模块。

        参数:
            weight_dict: dict, 各损失项名称及其对应权重
            losses: list, 需要计算的损失项名称列表
            saliency_margin: float, 显著性对比损失的 margin 值
            gamma: float, 困难样本挖掘的权重因子
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        self.gamma = gamma

    def loss_spans(self, outputs, targets):
        """基于 frame_pred 结果的 span 损失（旧版本）。

        注意: 此函数中存在梯度断裂问题（.item() 调用会脱离计算图），
        当前未在 get_loss 中使用，实际使用的是 new_loss_spans。

        输入:
            outputs: dict, 模型输出，包含 frame_pred 和 chunk_idx
            targets: dict, 目标信息，包含 meta
        输出:
            dict: {'loss_spans_v1': tensor}
        """
        frame_pred = outputs['frame_pred']  # (batch_size, short_memory_len, 4)
        chunk_idx = outputs.get('chunk_idx', None)

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs")

        device = frame_pred.device
        batch_size = frame_pred.shape[0]
        seq_len = frame_pred.shape[1]

        loss_v1 = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        for i, idx in enumerate(chunk_idx):
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue

            short_memory_start = target_meta['short_memory_start']
            start_prob = torch.sigmoid(frame_pred[i, :, 0])
            end_prob = torch.sigmoid(frame_pred[i, :, 2])

            # 枚举所有可能的 (start, end) span 并计算置信度
            span_scores = []
            for st in range(seq_len):
                for ed in range(st, seq_len):
                    conf = start_prob[st] * end_prob[ed]
                    global_st = st + short_memory_start
                    global_ed = ed + short_memory_start
                    span_scores.append((global_st, global_ed, conf))

            span_scores.sort(key=lambda x: x[2], reverse=True)
            top_k = min(5, len(span_scores))
            top_spans = span_scores[:top_k]

            gt_windows = target_meta['gt_windows']

            span_ious = []
            pred_confs = []
            for st, ed, conf in top_spans:
                pred_span = torch.tensor([[st, ed]], device=device)
                max_iou = float('-inf')

                for gt_start, gt_end in gt_windows:
                    gt_span = torch.tensor([[gt_start, gt_end]], device=device)
                    iou = generalized_temporal_iou(pred_span, gt_span)
                    max_iou = max(max_iou, iou.item())

                span_ious.append(max_iou)
                pred_confs.append(conf.item())

            if not span_ious:
                continue

            span_ious = torch.tensor(span_ious, device=device, requires_grad=True)
            pred_confs = torch.tensor(pred_confs, device=device, requires_grad=True)

            loss_v1 = loss_v1 + F.mse_loss(pred_confs, span_ious)
            valid_samples += 1

        if valid_samples > 0:
            loss_v1 = loss_v1 / valid_samples

        return {
            'loss_spans_v1': loss_v1
        }

    def new_loss_spans(self, outputs, targets):
        """基于 pred_spans 的 span 损失，使用 L1 距离和 GIoU。

        输入:
            outputs: dict, 模型输出，包含 pred_spans 和 chunk_idx
            targets: dict, 目标信息，包含 meta
        输出:
            dict: {'loss_span': tensor, 'loss_giou': tensor}
        """
        pred_spans = outputs['pred_spans']  # (batch_size, num_queries, 2)
        chunk_idx = outputs.get('chunk_idx', None)

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs")

        device = pred_spans.device
        batch_size = pred_spans.shape[0]
        num_queries = pred_spans.shape[1]

        loss_span = torch.tensor(0.0, device=device, requires_grad=True)
        loss_giou = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        for i, idx in enumerate(chunk_idx):
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue

            gt_spans = target_meta['gt_windows']
            if not gt_spans:
                continue

            gt_spans = torch.tensor(gt_spans, dtype=torch.float32, device=device)

            sample_pred_spans = pred_spans[i]  # (num_queries, 2)
            short_start = targets['short_memory_start']
            sample_pred_spans = span_cxw_to_xx(sample_pred_spans)
            sample_pred_spans = sample_pred_spans * target_meta["start_label"].shape[0]
            sample_pred_spans = sample_pred_spans + torch.tensor(short_start[i], dtype=torch.float32, device=device)

            # 计算所有预测和真实片段对的 L1 距离
            l1_dist = torch.cdist(sample_pred_spans, gt_spans, p=1)
            min_l1_loss = l1_dist.min(dim=1)[0].mean()

            # 计算广义 IoU
            giou_matrix = 1 - generalized_temporal_iou(sample_pred_spans, gt_spans)
            max_giou_per_query = giou_matrix.max(dim=1)[0]
            mean_max_giou = max_giou_per_query.mean()

            loss_span = loss_span + min_l1_loss
            loss_giou = loss_giou + mean_max_giou
            valid_samples += 1

        if valid_samples > 0:
            loss_span = loss_span / valid_samples
            loss_giou = loss_giou / valid_samples

        return {
            'loss_span': loss_span / 40,  # 缩放使 loss_span 保持在 1 附近
            'loss_giou': loss_giou
        }

    def loss_labels(self, outputs, targets, log=True):
        """帧分类损失函数，先筛选有效样本，再批量计算 BCE 损失。

        输入:
            outputs: dict, 模型输出，包含 frame_pred 和 chunk_idx
            targets: dict, 目标信息，包含 meta（start_label, middle_label, end_label）
            log: bool, 是否记录日志（当前未使用）
        输出:
            dict: 包含 loss_start_label, loss_middle_label, loss_end_label, loss_label
        """
        frame_pred = outputs['frame_pred']  # (batch_size, short_memory_len, 4)
        chunk_idx = outputs.get('chunk_idx', None)

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # 筛选有效样本
        valid_preds = []
        valid_start_labels = []
        valid_middle_labels = []
        valid_end_labels = []

        for i, idx in enumerate(chunk_idx):
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue

            start_label = target_meta['start_label']
            middle_label = target_meta['middle_label']
            end_label = target_meta['end_label']

            pred = frame_pred[i]  # (short_memory_len, 4)

            valid_preds.append(pred)
            valid_start_labels.append(start_label)
            valid_middle_labels.append(middle_label)
            valid_end_labels.append(end_label)

        if not valid_preds:
            return {"loss_label": torch.tensor(0.0, device=frame_pred.device)}

        # 将有效样本组织成批量张量
        valid_preds = torch.stack(valid_preds)
        valid_start_labels = torch.stack(valid_start_labels).to(valid_preds.device)
        valid_middle_labels = torch.stack(valid_middle_labels).to(valid_preds.device)
        valid_end_labels = torch.stack(valid_end_labels).to(valid_preds.device)

        # 批量计算损失
        losses = {}
        total_loss = 0

        label_types = ['start_label', 'middle_label', 'end_label']
        label_tensors = [valid_start_labels, valid_middle_labels, valid_end_labels]
        pred_indices = [0, 1, 2]

        for label_type, target, pred_idx in zip(label_types, label_tensors, pred_indices):
            pred_slice = valid_preds[:, :, pred_idx]  # (num_valid_samples, num_frames)
            pred_probs = torch.sigmoid(pred_slice)

            assert not torch.isnan(pred_probs).any(), "pred_probs contains NaN"
            assert not torch.isinf(pred_probs).any(), "pred_probs contains inf"
            assert (pred_probs >= 0).all() and (pred_probs <= 1).all(), "pred_probs must be in [0, 1]"

            # 计算 BCE 损失
            loss = F.binary_cross_entropy(
                pred_probs.to(torch.float64), target, reduction='none'
            )

            # # 备选: 使用 BCE with logits（跳过手动 sigmoid）
            # loss = F.binary_cross_entropy_with_logits(
            #     pred_slice, target, reduction='none'
            # )

            # 动态权重：赋予较难分类的样本更高权重
            weight = torch.abs(pred_probs - target)
            weight = torch.pow(weight, self.gamma)
            loss = (loss * weight).mean()

            if torch.isnan(loss).any():
                print(f"WARNING: NaN detected in {label_type} loss. Target: {target}, Pred: {pred_slice}")

            losses[f'loss_{label_type}'] = loss
            total_loss += loss

        losses['loss_label'] = total_loss

        return losses

    def loss_saliency(self, outputs, targets, log=True):
        """显著性损失函数，先筛选有效样本，再批量计算损失。

        包含三种损失组件（当前仅使用对比损失）:
            1. 对比损失: 正样本分数应高于负样本
            2. BCE 损失: 二分类交叉熵
            3. 平滑 L1 损失

        输入:
            outputs: dict, 模型输出，包含 saliency_scores 和 chunk_idx
            targets: dict, 目标信息，包含 saliency 相关标签
            log: bool, 是否记录日志（当前未使用）
        输出:
            dict: {'loss_saliency': tensor}
        """
        if ("saliency_scores" not in outputs or
            "saliency_pos_labels" not in targets or
            "short_memory_start" not in targets):
            print("info lacked to compute loss saliency")
            return {"loss_saliency": 0}

        saliency_scores = outputs["saliency_scores"]  # (batch_size, num_frames)
        chunk_idx = outputs.get('chunk_idx', None)

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # 筛选有效样本
        valid_scores = []
        valid_pos_labels = []
        valid_neg_labels = []
        valid_tgt_scores = []

        for i, idx in enumerate(chunk_idx):
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue

            pos_labels = target_meta['saliency_pos_labels']
            neg_labels = target_meta['saliency_neg_labels']
            if -1 in pos_labels or -1 in neg_labels:
                continue

            pred_scores = saliency_scores[i]
            short_memory_start = target_meta['short_memory_start']
            tgt_scores = torch.tensor(
                target_meta['saliency_all_labels'], dtype=torch.float32, device=pred_scores.device
            )

            # 转换为相对于 short memory 的局部索引
            pos_indices = [pos_idx - short_memory_start for pos_idx in pos_labels]
            neg_indices = [neg_idx - short_memory_start for neg_idx in neg_labels]

            if not pos_indices or not neg_indices:
                continue

            pos_indices = torch.tensor(pos_indices, dtype=torch.long, device=pred_scores.device)
            neg_indices = torch.tensor(neg_indices, dtype=torch.long, device=pred_scores.device)

            valid_scores.append(pred_scores)
            valid_tgt_scores.append(tgt_scores)
            valid_pos_labels.append(pos_indices)
            valid_neg_labels.append(neg_indices)

        if not valid_scores:
            return {"loss_saliency": 0}

        valid_scores = torch.stack(valid_scores)  # (num_valid_samples, num_frames)
        valid_tgt_scores = torch.stack(valid_tgt_scores)

        # 1. 对比损失
        base_losses = []
        for i in range(len(valid_scores)):
            pos_indices = valid_pos_labels[i]
            neg_indices = valid_neg_labels[i]

            pos_scores = valid_scores[i, pos_indices]
            neg_scores = valid_scores[i, neg_indices]

            base_loss = torch.clamp(
                self.saliency_margin + neg_scores - pos_scores,
                min=0
            ).mean()

            base_losses.append(base_loss)

        loss_contrastive = torch.mean(torch.stack(base_losses))

        # 2. BCE 损失（当前未使用，保留供消融实验）
        binary_targets = (valid_tgt_scores > 0).float()
        pred_probs = torch.sigmoid(valid_scores)
        ce_loss_fn = torch.nn.BCELoss()
        loss_ce = ce_loss_fn(pred_probs, binary_targets)

        # 3. 平滑 L1 损失（当前未使用，保留供消融实验）
        smooth_l1_loss_fn = torch.nn.SmoothL1Loss()
        loss_smooth_l1 = smooth_l1_loss_fn(valid_scores, valid_tgt_scores)

        # 当前仅使用对比损失（可切换为组合损失: loss_contrastive + loss_ce + loss_smooth_l1）
        final_loss = loss_contrastive

        return {"loss_saliency": final_loss}

    def loss_classifier(self, outputs, targets, **kwargs):
        """分类器损失，用于判断长期记忆中是否包含足够的 gt 帧。

        注意: 当前 forward 中未输出 classifier_logits，此损失项未启用。

        输入:
            outputs: dict, 模型输出，需包含 classifier_logits
            targets: dict, 需包含 gt_frames_in_long_memory
        输出:
            dict: {'loss_classifier': tensor}
        """
        classifier_logits = outputs['classifier_logits']
        classifier_labels = (targets['gt_frames_in_long_memory'] > 16).float()
        loss = F.binary_cross_entropy_with_logits(classifier_logits, classifier_labels)

        return {"loss_classifier": loss}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        """根据损失名称分发到对应的损失计算函数。

        输入:
            loss: str, 损失名称
            outputs: dict, 模型输出
            targets: dict, 目标信息
            indices: 匹配索引（当前未使用）
        输出:
            dict, 对应损失项的字典
        """
        loss_map = {
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "span": self.new_loss_spans,
            "classifier": self.loss_classifier,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    @countTime
    def forward(self, outputs, targets):
        """计算所有损失并返回总损失。

        输入:
            outputs: dict, 模型输出字典
            targets: dict, 目标字典
        输出:
            (total_loss, losses): 总损失标量和各项损失的字典
        """
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices=None))

        # 当前直接累加所有损失（可切换为加权求和: sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())）
        total_loss = sum(losses.values())

        return total_loss, losses


class MLP(nn.Module):
    """简单的多层感知机（前馈网络）。

    输入:
        input_dim: int, 输入维度
        hidden_dim: int, 隐藏层维度
        output_dim: int, 输出维度
        num_layers: int, 层数
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """前向传播，除最后一层外都使用 ReLU 激活。"""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """可配置 LayerNorm、Dropout 和 ReLU 的线性层。

    输入:
        in_hsz: int, 输入维度
        out_hsz: int, 输出维度
        layer_norm: bool, 是否使用 LayerNorm
        dropout: float, dropout 率
        relu: bool, 是否使用 ReLU 激活
    """

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播: (N, L, D) -> (N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    """构建在线 Moment-DETR 模型和损失函数。

    输入:
        args: 包含模型配置参数的命名空间对象
    输出:
        (model, criterion): 模型实例和损失函数实例
    """
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = OLMomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        compress_len=args.compress_len,
        use_vid_compression=args.use_vid_compression,
        use_inter_memory=args.use_inter_memory,
        weight_alpha=args.weight_alpha,
        attn_weight=args.attn_weight,
        future_memory_sample_len=args.future_memory_sample_len,
        memory_topk=args.memory_topk,
        memory_slots=getattr(args, "memory_slots", 64),
        ablate_delta_stream=getattr(args, "ablate_delta_stream", False),
        use_saliency_bias=getattr(args, "use_saliency_bias", True),
        use_distance_bias=getattr(args, "use_distance_bias", True),
        slot_select_mode=getattr(args, "slot_select_mode", "header"),
        disable_phase_alignment=getattr(args, "disable_phase_alignment", False),
        fixed_phase_start=getattr(args, "fixed_phase_start", 0),
        inter_memory_use_diversity_loss=getattr(args, "inter_memory_use_diversity_loss", True),
        inter_memory_use_struct_loss=getattr(args, "inter_memory_use_struct_loss", True),
        inter_memory_diversity_loss_coef=getattr(args, "inter_memory_diversity_loss_coef", 0.001),
        inter_memory_struct_loss_coef=getattr(args, "inter_memory_struct_loss_coef", 0.01),
        inter_memory_slot_len=getattr(args, "inter_memory_slot_len", None),
        inter_memory_legacy_v1=getattr(args, "inter_memory_legacy_v1", False),
    )

    matcher = build_matcher(args)

    weight_dict = {
        "loss_start_label": args.start_label_loss_coef,
        "loss_middle_label": args.middle_label_loss_coef,
        "loss_end_label": args.end_label_loss_coef,
        "loss_saliency": args.saliency_loss_coef,
        # "loss_spans": args.span_loss_coef,  # span 损失权重（当前未启用）
    }

    # losses = ['labels', 'saliency', 'span', 'classifier']  # 含分类器损失的版本
    losses = ['labels', 'saliency', 'span']

    # loss_label 消融实验
    # losses = ['labels', 'saliency']

    criterion = SetCriterionOl(
        weight_dict=weight_dict,
        losses=losses,
        saliency_margin=args.saliency_margin,
    )

    criterion.to(device)
    return model, criterion
