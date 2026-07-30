"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Copyright (c) 2024 EdenGabriel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2022 WonJun Moon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time

from lighthouse.common.compressor import SimpleCompressor
from lighthouse.common.compressor import (
    CrossAttentionResidualCompressor,
    CrossAttentionResidualCompressorWithQuery,
    ResidualCompressor,
)
from lighthouse.common.compressor import (
    CompressorWithExternalWeights,
    CompressorWithPositionWeights,
    CompressorWithPostWeighting,
    CompressorWithSmoothMechanism,
    CompressorWithWeightedKV,
)
from lighthouse.common.compressor import (
    ConcatCompressor,
    CrossAttentionConcatCompressor,
)
from lighthouse.common.compressor import (
    MultimodalTokenCompressor,
    TextGuidedCompressor,
    TextPresentGuidedCompressor,
)
from lighthouse.common.compressor import OnlineCrossAttentionResidualCompressor
from lighthouse.common.matcher import build_matcher
from lighthouse.common.memoryModule import InterMemorySeq
from lighthouse.common.misc import accuracy
from lighthouse.common.position_encoding import build_position_encoding
from lighthouse.common.taskweave_transformer import build_transformer
from lighthouse.common.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class TaskWeave(nn.Module):
    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, weight_alpha, attn_weight, future_memory_sample_len, aux_loss=False, max_v_l=75, span_loss_type="l1",
                 use_txt_pos=False, n_input_proj=2, aud_dim=0, 
                 mr2hd=False, compress_len=10, use_vid_compression=True, use_inter_memory=True, memory_topk=4,
                 compressor_output_mode="dual_intra", inter_memory_output_mode="future_only",
                 ablate_delta_stream=False, use_saliency_bias=True, use_distance_bias=True,
                 slot_select_mode="header", disable_phase_alignment=False, fixed_phase_start=0,
                 inter_memory_use_diversity_loss=True, inter_memory_use_struct_loss=True,
                 inter_memory_diversity_loss_coef=0.001, inter_memory_struct_loss_coef=0.01,
                 taskweave_hd_mr_epoch_thres=40):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         TaskWeave can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background

            online版本添加如下输入参数：
            weight_alpha、attn_weight：压缩器模块的权重参数，分别调整  融合外部权重  和  残差连接最近帧  时的融合比例
            future_memory_sample_len：未来帧采样长度，用来初始化intra-memory，以及判断是否进行未来信息预测
            compress_len：压缩器模块的压缩长度
            use_vid_compression：是否使用视频压缩
            use_inter_memory：是否使用跨帧记忆模块
            memory_topk：intra-memory模块的topk参数    
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
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        self.mr2hd = mr2hd

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
        self.hidden_dim = hidden_dim

        self.saliency_proj_linear = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj_local = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.output_saliency = nn.Linear(hidden_dim, 1)

        # Define variables with PyTorch
        self.hd_log_var = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mr_log_var = nn.Parameter(torch.zeros(1), requires_grad=True)
        nn.init.constant_(self.hd_log_var, 0.0)
        nn.init.constant_(self.mr_log_var, 0.0)

        # online 版本
        # 添加帧分类器，定义类属性

        self.frame_class_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 输出维度仍为4，但表示4个二分类的logits
        self.compress_len = compress_len
        self.future_memory_sample_len = future_memory_sample_len
        self.use_inter_memory = use_inter_memory
        self.memory_topk = memory_topk
        self.compressor_output_mode = compressor_output_mode
        self.inter_memory_output_mode = inter_memory_output_mode
        self.inter_memory_use_diversity_loss = inter_memory_use_diversity_loss
        self.inter_memory_use_struct_loss = inter_memory_use_struct_loss
        self.inter_memory_diversity_loss_coef = inter_memory_diversity_loss_coef
        self.inter_memory_struct_loss_coef = inter_memory_struct_loss_coef
        self.taskweave_hd_mr_epoch_thres = taskweave_hd_mr_epoch_thres

        # 一个字典，记录每个[qid_vid]对对应的视频的显著性分数值
        # 用saliency_score_dict[qid_vid]可以快速访问
        # saliency_score_dict结构：每个[qid_vid]索引两个元素：1.start：当前vid对于qid从start·开始采样；
        #                                              2.score：二维float数组，score[i][j]代表从start开始的第i帧的第j个预测saliency_score
        self.saliency_score_dict = {}

        # 添加视频压缩模块
        self.use_vid_compression = use_vid_compression
        if self.use_vid_compression:
            # self.memory_compressor = TextGuidedCompressor(t_dim = hidden_dim,v_dim = hidden_dim,hidden_dim = hidden_dim,compress_len=self.compress_len,weight_alpha=weight_alpha)
            # 效率评估修改点 3：follow tsgsv做了推理时优化
            self.memory_compressor = CrossAttentionResidualCompressorWithQuery(
                dimension=hidden_dim,
                compress_len=self.compress_len,
                weight_alpha=weight_alpha,
                attn_weight=attn_weight,
                compressor_output_mode=self.compressor_output_mode,
                ablate_delta_stream=ablate_delta_stream,
                use_saliency_bias=use_saliency_bias,
                use_distance_bias=use_distance_bias,
            )
            # self.memory_compressor = OnlineCrossAttentionResidualCompressor(dimension=hidden_dim, compress_len=self.compress_len, weight_alpha=weight_alpha, attn_weight=attn_weight)
            

        # 添加inter-memory模块
        if self.use_inter_memory and self.future_memory_sample_len > 0:
            # 记忆槽长度为压缩后的帧特征长度，输入压缩后的历史帧特征序列
            default_slot_len = self.compress_len + future_memory_sample_len * 2
            self.inter_memory = InterMemorySeq(
                d_model=hidden_dim,
                future_length=future_memory_sample_len,
                memory_slots=64,
                slot_len=default_slot_len,
                compress_len=self.compress_len,
                similarity_threshold=0.9,
                topk=self.memory_topk,
                diversity_loss_weight=0.5,
                optimization_interval=1,
                optimizer_lr=1e-4,
                slot_select_mode=slot_select_mode,
                disable_phase_alignment=disable_phase_alignment,
                fixed_phase_start=fixed_phase_start,
            )



    def forward(self, src_txt, src_txt_mask, src_vid, memory_len, src_vid_mask, chunk_idx, qid_vid, short_memory_start, src_aud=None, src_aud_mask=None,
                long_memory_weight=None, gt_frames_in_long_memory=None, frame_index_long=None, frame_index_long_mask=None,
                center_index=None, delta_gt=None, delta_gt_valid=None, training_epoch=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.

            为online版本添加的参数：
            chunk_idx：每个样本的编号，用于在计算metric的时候对齐
            qid_vid：用于对齐
            short_memory_start：用于识别每个样本的起始时刻
            long_memory_weight：长期记忆的外部权重（显著性分数）
            gt_frames_in_long_memory：长期记忆中的ground truth帧
        """

        # #计时点t1：准备压缩之前的proj操作耗时
        # st = time.time()

        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        # 取长期记忆视频，长期记忆mask
        long_vid = src_vid[:, :memory_len[0]]
        long_vid_mask = src_vid_mask[:, :memory_len[0]]

        present_vid = src_vid[:, memory_len[0]:]
        memory_struct_loss = None

        # ed = time.time()
        # with open("eff_time_static.txt", "a") as f:
        #     f.write(f"t1:准备压缩之前的proj操作耗时:{ed - st}\n")   

        # for online setting
        # 将长期记忆压缩
        if self.use_vid_compression:
            # 效率评估修改点 5：在推理阶段第一次调用时启用在线模式
            # if not self.training and not getattr(self.memory_compressor, "online_mode", False):
            #     batch_size = src_vid.size(0)
            #     device = src_vid.device
            #     max_history_len = 64  # 或者根据你希望缓存的历史帧长度
            #     self.memory_compressor.enable_online_mode(max_history_len, batch_size, device)

            # # 计时点4-5：为intra-memory准备extern_weight的步骤耗时
            # st = time.time()

            # 效率评估修改点 6：infer的时候也不加载 / 保存extern_weight了
            # 全都set none
            # extern_weight = None

            # 如果是train，使用数据集提供的显著性分数作为extern_weight
            if self.training:
                extern_weight = long_memory_weight
            # inference时，使用过去推理的结果
            else:
                all_weights=[]
                extern_weight = None

                for batch_idx in range(len(qid_vid)):
                    output = self.get_long_memory_weight_by_qid_vid(qid_vid[batch_idx][0], qid_vid[batch_idx][1])

                    if output is None:
                        # 清空已收集的权重，确保一致性
                        all_weights = []  
                        extern_weight = None
                        break
                    indexed_weight = self.index_weight(output[1], output[0], short_memory_start[batch_idx], memory_len[0])
                    all_weights.append(torch.from_numpy(indexed_weight).float())
                
                # 只有在所有batch都成功获取权重时才stack
                if len(all_weights) == len(qid_vid):
                    extern_weight = torch.stack(all_weights, dim=0)
                else:
                    extern_weight = None 
                    
            # 确保extern_weight在和long_vid在相同的设备上
            if extern_weight is not None:
                extern_weight = extern_weight.to(long_vid.device)

            # 检查压缩前的long_vid是否包含NaN
            if torch.isnan(long_vid).any():
                print("Before compression, long_vid contains NaN!")

            # ed = time.time()
            # with open("eff_time_static.txt", "a") as f:
            #     f.write(f"4-5:为intra-memory准备extern_weight的步骤耗时:{ed - st}\n")  
            
            # # 计时点5：intra-memory模块耗时
            # st = time.time()

            # 压缩长期记忆
            compress_long_res, compress_long_mask = self.memory_compressor(
                Fv=long_vid,
                vid_mask=long_vid_mask,
                Ft=src_txt,
                text_mask=src_txt_mask,
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
            

            # 检查压缩后的长期记忆是否存在NaN
            if torch.isnan(compress_long_vid).any():
                print("compress_long_vid contains NaN!")
                print("long_vid:", long_vid)
                print("src_txt:", src_txt)
                print("src_txt_mask:", src_txt_mask)
                print("long_vid_mask:", long_vid_mask)
                print("extern_weight:", extern_weight)
                input("Press Enter to continue...")

            #   将压缩后的长期记忆和短期，future记忆拼接得到新的src_vid, src_vid_mask
            src_vid, src_vid_mask = torch.cat([compress_long_vid, src_vid[:, memory_len[0]:]], dim=1), \
                torch.cat([compress_long_mask, src_vid_mask[:, memory_len[0]:]], dim=1)
            memory_len[0] = self.compress_len

            # ed = time.time()
            # with open("eff_time_static.txt", "a") as f:
            #     f.write(f"5:intra-memory耗时+拼接src:{ed - st}\n")

        # 使用压缩后的历史记忆维护interMemory
        if self.use_inter_memory and self.inter_memory is not None and self.use_vid_compression:
            delta_gt_for_memory = None
            delta_gt_valid_for_memory = None
            if delta_gt is not None:
                delta_gt_for_memory = delta_gt.to(src_vid.device)
                if delta_gt_for_memory.shape[-1] != present_vid.shape[-1]:
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

            # # 计时点6：inter-memory模块耗时
            # st = time.time()
            
            # 确保 InterMemory 的参数与输入同设备
            if self.inter_memory.memory.device != src_vid.device:
                self.inter_memory.memory.data = self.inter_memory.memory.data.to(src_vid.device)
                self.inter_memory.future_embedding.data = self.inter_memory.future_embedding.data.to(src_vid.device)
                self.inter_memory.header_tokens.data = self.inter_memory.header_tokens.data.to(src_vid.device)

            # 更新interMemory
            future_vid, future_mask = self.inter_memory.forward_with_controls(
                history_features=compress_long_vid,
                mask=compress_long_mask,
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
            if future_vid.shape[1] > self.future_memory_sample_len:
                future_vid = future_vid[:, -self.future_memory_sample_len:]
                future_mask = future_mask[:, -self.future_memory_sample_len:]

            # 将未来部分拼接
            src_vid = torch.cat([src_vid, future_vid], dim=1)
            src_vid_mask = torch.cat([src_vid_mask, future_mask], dim=1)

            # ed = time.time()
            # with open("eff_time_static.txt", "a") as f:
            #     f.write(f"6:inter-memory耗时:{ed - st}\n")
        
        # # 计时点7：统计base model部分耗时
        # st = time.time()

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        tmp_mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        video_length = src_vid.shape[1]

        # ============================= hd2mr ============================= 
        epoch_thres = self.taskweave_hd_mr_epoch_thres
        short_start = memory_len[0]
        short_end = short_start + memory_len[1]

        if training_epoch is None or training_epoch + 1 > epoch_thres:
            _, _, _, hd_memory = self.transformer(src, ~tmp_mask, self.query_embed.weight, pos, video_length=video_length)

            pos_global_local = self.saliency_proj_local(self.saliency_proj_linear(hd_memory).transpose(1, 2)).transpose(1, 2)
            saliency_scores = self.output_saliency(pos_global_local).squeeze(-1) + torch.sum(hd_memory,dim=-1).squeeze(-1)/ np.sqrt(self.hidden_dim)
            saliency_scores = saliency_scores.unsqueeze(1)
            batch_means = torch.mean(saliency_scores, dim=2, keepdim=True)  
            saliency_mask = torch.where(saliency_scores > batch_means, torch.ones_like(saliency_scores), torch.zeros_like(saliency_scores))
            saliency_mask = saliency_mask.permute(0,2,1) 
            saliency_mask = saliency_mask.expand(saliency_mask.shape[0],video_length,src_vid.shape[-1])
            
            # update src
            src[:,:video_length] = src[:,:video_length] + src[:,:video_length]*saliency_mask

            hs, reference, mr_memory, hd_memory = self.transformer(src, ~tmp_mask, self.query_embed.weight, pos, video_length=video_length)
            # hs: torch.Size([2, 32, 10, 256]) # layers, bsz, queries, dim
            outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.span_embed(hs)
            outputs_coord = tmp + reference_before_sigmoid
            if self.span_loss_type == "l1":
                outputs_coord = outputs_coord.sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
            if memory_struct_loss is not None:
                out["loss_memory_struct"] = memory_struct_loss
            out["saliency_scores"] = saliency_scores.squeeze(1)
            out["saliency_scores"] = out["saliency_scores"][:, short_start:short_end]

            # 这里纠正了之前版本的一个小问题
            # 之前修改taskWeave的代码的时候，顺着原来的思路（mr帧预测头使用和hd预测头一样的特征）
            # 但是taskWeave中，修改后的trasformer是为mr任务提供了专门的memory的，强行用pos_global_local（hd_memory初步处理过的特征）不仅在mr上效果不好，还可能给hd任务带来错误的反向传播信号
            vid_mem = mr_memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
            out['frame_pred'] = self.frame_class_embed(vid_mem)[:, short_start:short_end]    # (bsz, short_memory_length, 4)
            
            
        else:
            hs, reference, mr_memory, hd_memory = self.transformer(src, ~tmp_mask, self.query_embed.weight, pos, video_length=video_length)
            # hs: torch.Size([2, 32, 10, 256]) # layers, bsz, queries, dim
            outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.span_embed(hs)
            outputs_coord = tmp + reference_before_sigmoid
            if self.span_loss_type == "l1":
                outputs_coord = outputs_coord.sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
            if memory_struct_loss is not None:
                out["loss_memory_struct"] = memory_struct_loss

            pos_global_local = self.saliency_proj_local(self.saliency_proj_linear(hd_memory).transpose(1, 2)).transpose(1, 2)
            out["saliency_scores"] = self.output_saliency(pos_global_local).squeeze(-1) + torch.sum(hd_memory,dim=-1).squeeze(-1)/ np.sqrt(self.hidden_dim)
            out["saliency_scores"] = out["saliency_scores"][:, short_start:short_end]
            
            # 添加帧的类别预测
            vid_mem = mr_memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
            out['frame_pred'] = self.frame_class_embed(vid_mem)[:, short_start:short_end]    # (bsz, short_memory_length, 4)

        # 添加 chunk_idx 到输出
        out["chunk_idx"] = chunk_idx  # [batch_size]

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()
        pos_neg = pos.clone()  # since it does not use actual content
        _, _, mr_memory_neg, hd_memory_neg= self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length)
        vid_mem_neg = mr_memory_neg[:, :src_vid.shape[1]]

        neg_global_local = self.saliency_proj_local(self.saliency_proj_linear(hd_memory_neg).transpose(1, 2)).transpose(1, 2)
        out["saliency_scores_neg"] = self.output_saliency(neg_global_local).squeeze(-1) + torch.sum(hd_memory_neg,dim=-1).squeeze(-1)/ np.sqrt(self.hidden_dim)
        out["saliency_scores_neg"] = out["saliency_scores_neg"][:, short_start:short_end]

        txt_mem = mr_memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = mr_memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        out["video_mask"] = src_vid_mask[:, short_start:short_end]
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # ed = time.time()
        # with open("eff_time_static.txt", "a") as f:
        #     f.write(f"7:base model耗时:{ed - st}\n")

        # # 计时点7-1：统计维护和更新saliency的逻辑耗时
        # st = time.time()

        # 效率评估修改点 6：infer的时候也不加载 / 保存extern_weight了

        # 维护和更新saliency_score_dict的逻辑
        if self.use_vid_compression and not self.training and qid_vid is not None and short_memory_start is not None:
            qid_vid_list = qid_vid # 转换为Python列表
            short_memory_start_list = short_memory_start
            batch_size = src_vid.shape[0]

            for i in range(batch_size):
                qid,vid = qid_vid_list[i]
                current_short_start = short_memory_start_list[i]
                current_saliency_score = out["saliency_scores"][i, :]
                current_length = current_saliency_score.size(0)
                current_saliency_score_np = current_saliency_score.cpu().detach().numpy()

                weight_col = 1 if self.training else memory_len[1]
                if (qid,vid) not in self.saliency_score_dict:
                    # 初始化新条目
                    weight = np.full((current_length, weight_col), np.nan)
                    # 将当前预测结果填充到weight中
                    weight[:current_length, 0] = current_saliency_score_np
                    self.saliency_score_dict[(qid,vid)] = {
                        'start': current_short_start,
                        'weight': weight
                    }
                else:
                    entry = self.saliency_score_dict[(qid,vid)]
                    existing_start = entry['start']
                    existing_weight = entry['weight']
                    new_L = current_short_start - existing_start + current_length
                    new_weight = np.full((new_L, weight_col), np.nan)
                    # 将existing_weight复制到new_weight
                    new_weight[:existing_weight.shape[0], :weight_col] = existing_weight
                    # 将现有数据复制到新数组
                    for i, saliency_score in enumerate(current_saliency_score_np):
                        for insert_idx, latest_label in enumerate(new_weight[current_short_start + i - existing_start]):
                            if np.isnan(latest_label) :

                                if np.isnan(saliency_score):
                                    input("nan occur in saliency score")

                                new_weight[current_short_start + i - existing_start, insert_idx] = saliency_score
                                break
                    # 更新条目
                    entry['weight'] = new_weight
        # et = time.time()
        # with open("eff_time_static.txt", "a") as f:
        #     f.write(f"7-1：统计维护和更新saliency的逻辑耗时:{et - st}\n")

        out["hd_log_var"] = self.hd_log_var
        out["mr_log_var"] = self.mr_log_var
        return out

    def reset_saliency_score_dict(self):
        self.saliency_score_dict = {}

    # 重新初始化inter_memory，用于两个epoch之间
    def reset_inter_memory(self):
        if self.use_inter_memory and self.inter_memory is not None:
            self.inter_memory.reset()

    def process_probs_by_avg(self,probs):
        return np.mean(probs, axis=0)

    # 通过saliency_score_dict获取long memory的weight
    # 返回值:1.startl2.从start开始的全部的long_memory_weight,形状为(batch_size, long_memory_length)
    def get_long_memory_weight_by_qid_vid(self,qid,vid):
        if (qid,vid) not in self.saliency_score_dict:
            return None
        dict = self.saliency_score_dict[(qid,vid)]
        dict_weight = dict["weight"]
        res = []
        for frame in dict_weight:
            # 对于frame中的prob，先筛除nan
            valid_frame = frame[~np.isnan(frame)]

            if np.isnan(valid_frame).any():
                input("all nan occur in valid_frame")

            frame_weight = self.process_probs_by_avg(valid_frame)
            # 对于每一帧的权重，归一化到0-1
            # 如果是qv数据集，先把分数缩放
            if ".0" in vid:
                frame_weight = frame_weight / 4
            if frame_weight > 1:
                frame_weight = 1
            if frame_weight < 0:
                frame_weight = 0
                
            res.append(frame_weight)
        return dict["start"], np.array(res)
    # 从get_long_memory_weight_by_qid_vid返回的weight_list中获取[st,ed]之间的分数
    # 返回的结果应该是pad到return_len的长度
    def index_weight(self,weight_list,weight_start,short_start,return_len):
        # 从weight_list中，取出[weight_start,short_start]之间的分数（不包括short_start对应的帧）作为raw_res
        # 如果raw_res比return_len小，有效值在低位索引处，用0填充
        # 如果raw_res比return_len大，则从后向前阶段，确保返回值长度
        weight_list = np.pad(weight_list, (weight_start, 0), 'constant', constant_values=0)
        if weight_start > short_start:
            return None
        raw_res = weight_list[:short_start]
        # 对应一些特殊情况，比如一些很长的视频，到最后只能将这些视频的相邻样本放进一个batch
        # 这时一些样本在推理时，并不是所有历史帧的分数都已经保存
        if len(weight_list)<short_start:
            raw_res = np.pad(weight_list, (0, short_start - len(weight_list)), 'constant', constant_values=0)
        if len(raw_res) < return_len:
            # 和mask保持一致，有效位在左侧
            raw_res = np.pad(raw_res, (0, return_len - len(raw_res)), 'constant', constant_values=0)
        else:
            raw_res = raw_res[-return_len:]
        return raw_res


class SetCriterionOl(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, saliency_margin=1, gamma=.0):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        self.gamma = gamma

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def new_loss_spans(self, outputs, targets):
        pred_spans = outputs['pred_spans']  # (batch_size, num_queries, 2)
        chunk_idx = outputs.get('chunk_idx', None)

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs")

        device = pred_spans.device
        batch_size = pred_spans.shape[0]
        num_queries = pred_spans.shape[1]

        # 初始化损失
        loss_span = torch.tensor(0.0, device=device, requires_grad=True)
        loss_giou = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        # 遍历每个样本
        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue

            # 获取真实的时间片段 (center_x, width)
            gt_spans = target_meta['gt_windows']  # list of [st, ed]
            if not gt_spans:
                continue  # 如果没有真实片段，跳过

            # 将真实片段转换为张量
            gt_spans = torch.tensor(gt_spans, dtype=torch.float32, device=device)  # (num_gt_spans, 2)

            # 当前样本的预测片段
            sample_pred_spans = pred_spans[i]  # (num_queries, 2)
            short_start = targets['short_memory_start']
            sample_pred_spans = span_cxw_to_xx(sample_pred_spans)
            sample_pred_spans = sample_pred_spans * target_meta["start_label"].shape[0]
            sample_pred_spans = sample_pred_spans + torch.tensor(short_start[i], dtype=torch.float32, device=device)

            # 计算所有预测和真实片段对的L1距离
            l1_dist = torch.cdist(sample_pred_spans, gt_spans, p=1)  # (num_queries, num_gt_spans)
            min_l1_loss = l1_dist.min(dim=1)[0].mean()  # 每个查询到最近真实片段的最小L1距离

            # 计算广义IoU
            giou_matrix = 1 - generalized_temporal_iou(sample_pred_spans, gt_spans)  # (num_queries, num_gt_spans)
            max_giou_per_query = giou_matrix.max(dim=1)[0]  # 每个查询的最大GIoU
            mean_max_giou = max_giou_per_query.mean()  # 所有查询的平均最大GIoU

            # 累加损失
            loss_span = loss_span + min_l1_loss
            loss_giou = loss_giou + mean_max_giou
            valid_samples += 1

        if valid_samples > 0:
            loss_span = loss_span / valid_samples
            loss_giou = loss_giou / valid_samples

        return {
            'loss_span': loss_span / 40,  # 将loss_span保持在1附近
            'loss_giou': loss_giou
        }

    def loss_labels(self, outputs, targets, log=True):
        """帧分类损失函数，先筛选有效样本，再批量计算损失"""
        frame_pred = outputs['frame_pred']  # (batch_size, short_memory_len, 4)
        chunk_idx = outputs.get('chunk_idx', None)  # 获取 chunk_idx

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # 筛选有效样本
        valid_preds = []
        valid_start_labels = []
        valid_middle_labels = []
        valid_end_labels = []
        
        # flag = 1
        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue  # 如果找不到对应的 target，跳过该样本

            # 获取当前样本的标签
            start_label = target_meta['start_label']  # (num_frames,)
            middle_label = target_meta['middle_label']  # (num_frames,)
            end_label = target_meta['end_label']  # (num_frames,)

            # 获取当前样本的预测
            pred = frame_pred[i]  # (short_memory_len, 4)

            # 添加到有效样本列表
            valid_preds.append(pred)
            valid_start_labels.append(start_label)
            valid_middle_labels.append(middle_label)
            valid_end_labels.append(end_label)

        if not valid_preds:
            return {"loss_label": torch.tensor(0.0, device=frame_pred.device)}

        # 将有效样本组织成批量张量
        valid_preds = torch.stack(valid_preds)  # (num_valid_samples, num_frames, 4)
        valid_start_labels = torch.stack(valid_start_labels)  # (num_valid_samples, num_frames)
        valid_middle_labels = torch.stack(valid_middle_labels)  # (num_valid_samples, num_frames)
        valid_end_labels = torch.stack(valid_end_labels)  # (num_valid_samples, num_frames)

        # 将标签移动到与预测相同的设备上
        valid_start_labels = valid_start_labels.to(valid_preds.device)
        valid_middle_labels = valid_middle_labels.to(valid_preds.device)
        valid_end_labels = valid_end_labels.to(valid_preds.device)

        # 批量计算损失
        losses = {}
        total_loss = 0

        label_types = ['start_label', 'middle_label', 'end_label']
        pred_indices = [0, 1, 2]  # pred 的索引对应 start, middle, end
        for label_type, pred_idx in zip(label_types, pred_indices):
            target = valid_start_labels if label_type == 'start_label' else \
                    valid_middle_labels if label_type == 'middle_label' else \
                    valid_end_labels  # (num_valid_samples, num_frames)
            pred_slice = valid_preds[:, :, pred_idx]  # (num_valid_samples, num_frames)

            # 将 logits 转换为概率值
            pred_probs = torch.sigmoid(pred_slice)  # (num_valid_samples, num_frames)

            # 检查 pred_probs 是否有效
            assert not torch.isnan(pred_probs).any(), "pred_probs contains NaN"
            assert not torch.isinf(pred_probs).any(), "pred_probs contains inf"
            assert (pred_probs >= 0).all() and (pred_probs <= 1).all(), "pred_probs must be in [0, 1]"

            # 计算加权BCE损失
            loss = F.binary_cross_entropy(
                pred_probs.to(torch.float64), target, reduction='none'
            )

            # # 计算加权BCE损失
            # loss = F.binary_cross_entropy_with_logits(
            #     pred_slice, target, reduction='none'
            # )

            # 动态权重，赋予较难的类别较高的权重
            weight = torch.abs(pred_probs - target)  # 使用 gamma 计算权重
            weight = torch.pow(weight, self.gamma)
            loss = (loss * weight).mean()
            # loss = loss.mean()  # 先不用weight

            if torch.isnan(loss).any():
                print(f"NaN detected in {label_type} loss. Target: {target}, Pred: {pred_slice}")
                input("Press Enter to continue...")

            losses[f'loss_{label_type}'] = loss
            total_loss += loss

        # 平均损失
        losses['loss_label'] = total_loss

        return losses

    def loss_saliency(self, outputs, targets, log=True):
        """在线版本的显著性损失计算，保留排序对比损失和负样本对损失"""
        if ("saliency_scores" not in outputs or 
            "saliency_pos_labels" not in targets or 
            "short_memory_start" not in targets):
            print("info lacked to compute loss saliency")
            return {"loss_saliency": 0}

        saliency_scores = outputs["saliency_scores"]  # (batch_size, num_frames)
        chunk_idx = outputs.get('chunk_idx', None)  # 获取 chunk_idx

        if chunk_idx is None:
            raise ValueError("chunk_idx is missing in outputs. Ensure it is passed correctly.")

        # 筛选有效样本
        valid_scores = []
        valid_pos_labels = []
        valid_neg_labels = []
        valid_tgt_scores = []
        valid_masks = []
        valid_neg_scores = []

        for i, idx in enumerate(chunk_idx):
            # 找到 targets 中对应的样本
            target_meta = next((meta for meta in targets['meta'] if meta['chunk_idx'] == idx.item()), None)
            if target_meta is None:
                continue  # 如果找不到对应的 target，跳过该样本

            # 检查 saliency_pos_labels 和 saliency_neg_labels 是否有效
            pos_labels = target_meta['saliency_pos_labels']
            neg_labels = target_meta['saliency_neg_labels']
            if -1 in pos_labels or -1 in neg_labels:
                continue  # 如果其中一个为 -1，跳过该样本

            # 获取当前样本的预测和标签
            pred_scores = saliency_scores[i]  # (num_frames,)
            short_memory_start = target_meta['short_memory_start']
            tgt_scores = torch.tensor(target_meta['saliency_all_labels'], dtype=torch.float32, device=pred_scores.device)
            
            # 获取视频掩码（如果存在）
            if "video_mask" in outputs:
                vid_mask = outputs["video_mask"][i]  # (num_frames,)
            else:
                vid_mask = torch.ones_like(pred_scores)  # 如果没有掩码，创建全1掩码

            # 转换为相对于 short memory 的局部索引
            pos_indices = torch.tensor([pos_idx - short_memory_start for pos_idx in pos_labels], 
                                    dtype=torch.long, device=pred_scores.device)
            neg_indices = torch.tensor([neg_idx - short_memory_start for neg_idx in neg_labels], 
                                    dtype=torch.long, device=pred_scores.device)

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue  # 如果没有有效索引，跳过该样本

            # 获取负样本分数（如果存在）
            if "saliency_scores_neg" in outputs:
                neg_scores = outputs["saliency_scores_neg"][i]  # (num_frames,)
            else:
                neg_scores = torch.zeros_like(pred_scores)  # 如果没有负样本分数，创建零张量

            # 添加到有效样本列表
            valid_scores.append(pred_scores)
            valid_tgt_scores.append(tgt_scores)
            valid_pos_labels.append(pos_indices)
            valid_neg_labels.append(neg_indices)
            valid_masks.append(vid_mask[-8:])
            valid_neg_scores.append(neg_scores)

        if not valid_scores:
            return {"loss_saliency": 0}

        # 将有效样本组织成批量张量
        valid_scores = torch.stack(valid_scores)  # (num_valid_samples, num_frames)
        valid_tgt_scores = torch.stack(valid_tgt_scores)  # (num_valid_samples, num_frames)
        valid_masks = torch.stack(valid_masks)  # (num_valid_samples, num_frames)
        valid_neg_scores = torch.stack(valid_neg_scores)  # (num_valid_samples, num_frames)

        # 1. 计算负样本对损失
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(valid_neg_scores)) * valid_masks).sum(dim=1).mean()

        # 2. 计算排序对比损失
        # 准备对比学习的数据
        saliency_scores_combined = torch.cat([valid_scores, valid_neg_scores], dim=1)  # (N, 2*L)
        saliency_labels_combined = torch.cat([valid_tgt_scores, torch.zeros_like(valid_tgt_scores)], dim=1)  # (N, 2*L)
        vid_token_mask = valid_masks.repeat([1, 2])  # (N, 2*L)
        
        # 应用掩码
        saliency_scores_combined = vid_token_mask * saliency_scores_combined + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # 排序对比损失计算
        for rand_idx in range(1, 12):
            drop_mask = (~(saliency_labels_combined > 100)).float()  # no drop
            
            pos_mask = (saliency_labels_combined >= rand_idx).float()  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores_combined * drop_mask / tau + (1. - drop_mask) * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask
            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        # 3. 计算基础对比损失
        base_losses = []
        for i in range(len(valid_scores)):
            pos_indices = valid_pos_labels[i]  # 可能是多个索引的 tensor
            neg_indices = valid_neg_labels[i]  # 可能是多个索引的 tensor

            pos_scores = valid_scores[i, pos_indices]  
            neg_scores = valid_scores[i, neg_indices]  

            # 基础对比损失
            base_loss = torch.clamp(
                self.saliency_margin + neg_scores - pos_scores, 
                min=0
            ).mean()

            base_losses.append(base_loss)

        # 平均所有样本的损失
        loss_contrastive = torch.mean(torch.stack(base_losses))

        # 4. 计算CE损失
        # 创建二分类标签：valid_tgt_scores > 0 为正样本，否则为负样本
        binary_targets = (valid_tgt_scores > 0).float()

        # 使用 sigmoid 函数将分数转换为概率
        pred_probs = torch.sigmoid(valid_scores)
        
        # 计算二元交叉熵损失
        ce_loss_fn = torch.nn.BCELoss()
        loss_ce = ce_loss_fn(pred_probs, binary_targets)

        # 5. 计算平滑L1损失
        # 使用 smooth L1 损失 (Huber loss)
        smooth_l1_loss_fn = torch.nn.SmoothL1Loss()
        loss_smooth_l1 = smooth_l1_loss_fn(valid_scores, valid_tgt_scores)

        # 合并所有损失
        # final_loss = loss_contrastive + loss_rank_contrastive + loss_neg_pair + loss_ce + loss_smooth_l1
        final_loss = loss_contrastive + loss_rank_contrastive + loss_neg_pair
        
        return {"loss_saliency": final_loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            "spans": self.new_loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        total_loss = sum(losses.values())
        return total_loss, losses   


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

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
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = TaskWeave(
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
        mr2hd=args.mr2hd,
        compress_len=args.compress_len,
        use_vid_compression=args.use_vid_compression,
        use_inter_memory=args.use_inter_memory,
        weight_alpha=args.weight_alpha,
        attn_weight=args.attn_weight,
        future_memory_sample_len=args.future_memory_sample_len,
        memory_topk=args.memory_topk,
        compressor_output_mode=getattr(args, "compressor_output_mode", "dual_intra"),
        inter_memory_output_mode=getattr(args, "inter_memory_output_mode", "future_only"),
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
        taskweave_hd_mr_epoch_thres=getattr(args, "taskweave_hd_mr_epoch_thres", 40),
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
        
    # For tvsum dataset (youtube_highlight?)
    use_matcher = not (args.dset_name == 'tvsum')

    criterion = SetCriterionOl(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, saliency_margin=args.saliency_margin,
    )
    criterion.to(device)
    return model, criterion