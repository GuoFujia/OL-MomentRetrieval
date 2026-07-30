import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F

import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple

def countTime(func: Callable) -> Callable:
    """装饰器：记录函数开始、结束和耗时到文件"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # 获取类名（如果是类方法）
        class_name = ""
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__ + "."
        func_name = f"{class_name}{func.__name__}"

        # 硬编码日志路径
        log_path = "./time_log.txt"

        # 记录开始时间
        start_time = time.perf_counter()
        with open(log_path, "a") as f:
            f.write(f"[Start] {func_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 执行原函数
        result = func(*args, **kwargs)

        # 记录结束时间和耗时
        elapsed = time.perf_counter() - start_time
        with open(log_path, "a") as f:
            f.write(
                f"[End] {func_name} -> "
                f"耗时: {elapsed:.6f}s, "
                f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
        return result
    return wrapper

class TextGuidedVideoAttention(nn.Module):
    """
    基于文本引导的视频特征注意力模块
    
    输入:
        Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
        Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
        vid_mask: 视频掩码，形状为 (batch_size, vid_len)，True/1表示有效位置
        text_mask: 文本掩码，形状为 (batch_size, num_token)，True/1表示有效位置
        
    输出:
        Fv_enhanced: 增强后的视频特征，保持原始形状 (batch_size, vid_len, v_dim)
    """
    
    def __init__(self, t_dim, v_dim, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        初始化文本引导视频注意力模块
        
        参数:
            t_dim: 文本特征维度
            v_dim: 视频特征维度
            hidden_dim: 注意力机制内部隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout率
        """
        super(TextGuidedVideoAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "隐藏维度必须能被头数整除"
        
        # 投影层 - 视频特征作为query，文本特征作为key和value
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.t_key_proj = nn.Linear(t_dim, hidden_dim)
        self.t_val_proj = nn.Linear(t_dim, hidden_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(hidden_dim, v_dim)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(v_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(v_dim, 4 * v_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * v_dim, v_dim),
            nn.Dropout(dropout)
        )
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(v_dim)

        # --- 新增：用于缓存注意力分数 ---
        self.register_buffer('_cached_attn_scores', None, persistent=False) # 使用 register_buffer 确保它随模型移动设备

    def _compute_scores(self, q, k, head_dim):
        """计算注意力分数 Q*K^T / sqrt(d_k)"""
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        return scores

    def _apply_masks_to_scores(self, scores, text_mask=None):
        """对注意力分数应用掩码"""
        if text_mask is not None:
            # 扩展文本掩码以匹配注意力分数的形状
            # (batch_size, num_token) -> (batch_size, 1, 1, num_token)
            expanded_text_mask = text_mask.unsqueeze(1).unsqueeze(2)
            # 在不有效的文本位置应用大的负值
            scores = scores.masked_fill(~expanded_text_mask.bool(), -1e9)
        return scores

    def update_text_enhanced_attn_scores_cache_incremental(self, new_frame_feat, Ft, text_mask):
        """
        增量更新文本增强注意力分数缓存。

        此函数假定：
        1. `Ft` 和 `text_mask` 在一段时间内保持不变。
        2. 整体的视频特征 FIFO 队列在调用此函数前已经更新，
           即 `new_frame_feat` 是队列中的最新帧，而 `_cached_attn_scores`
           对应的是更新前的旧队列（比当前队列少一帧）。

        参数:
            new_frame_feat (torch.Tensor): 新加入帧的视觉特征，
                                           形状为 (batch_size, 1, v_dim)。
            Ft (torch.Tensor): 不变的文本特征，形状为 (batch_size, num_token, t_dim)。
            text_mask (torch.Tensor): 不变的文本掩码，形状为 (batch_size, num_token)。

        返回:
            torch.Tensor: 更新后的注意力分数缓存，
                          形状为 (batch_size, num_heads, current_vid_len, num_token)。
        """
        batch_size, _, v_dim = new_frame_feat.size()
        _, num_token, t_dim = Ft.size()

        # 1. 计算新帧的 Query 投影
        q_new = self.v_proj(new_frame_feat)  # (B, 1, H)
        q_new = q_new.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, 1, hd)

        # 2. 获取并计算 Key 投影 (这部分是复用的)
        k = self.t_key_proj(Ft)  # (B, nt, H)
        k = k.view(batch_size, num_token, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, nt, hd)

        # 3. 计算新帧与文本的注意力分数
        new_scores_part = self._compute_scores(q_new, k, self.head_dim)  # (B, nh, 1, nt)
        new_scores_part = self._apply_masks_to_scores(new_scores_part, text_mask=text_mask) # Apply mask to new part too

        # 4. 更新缓存
        if self._cached_attn_scores is not None and self._cached_attn_scores.size(2) > 0:
            # a. 移除最旧的分数 (对应于被弹出的帧)
            #    假设缓存按FIFO顺序排列，移除第一个
            updated_cached_scores = self._cached_attn_scores[:, :, 1:, :]  # (B, nh, old_vid_len-1, nt)
            # b. 将新分数附加到缓存末尾
            self._cached_attn_scores = torch.cat([updated_cached_scores, new_scores_part], dim=2)  # (B, nh, old_vid_len, nt)
            # 注意：这里的 old_vid_len 应该等于当前期望的 vid_len
        else:
            # 如果没有缓存或者缓存为空，则初始化为新分数
            self._cached_attn_scores = new_scores_part # (B, nh, 1, nt)

        return self._cached_attn_scores # 返回更新后的完整缓存

    def forward_with_cached_scores(self, Ft, Fv, cached_scores, vid_mask=None, text_mask=None):
        """
        使用预先计算并缓存的注意力分数进行前向传播。

        参数:
             Ft (torch.Tensor): 不变的文本特征，形状为 (batch_size, num_token, t_dim)。
             Fv (torch.Tensor): 当前完整的视频特征队列，形状为 (batch_size, vid_len, v_dim)。
             cached_scores (torch.Tensor): 预先计算好的注意力分数，形状为 (batch_size, num_heads, vid_len, num_token)。
             vid_mask (torch.Tensor, optional): 视频掩码，形状为 (batch_size, vid_len)。
             text_mask (torch.Tensor, optional): 文本掩码，形状为 (batch_size, num_token)。

         返回:
             torch.Tensor: 增强后的视频特征，形状为 (batch_size, vid_len, v_dim)。
        """
        # 注意：这里不再需要重新计算 scores，直接使用传入的 cached_scores
        # 但我们仍然需要计算 Value 并执行后续步骤
        batch_size, vid_len, v_dim = Fv.size()
        _, num_token, _ = Ft.size()

        # 1. 计算 Value 投影 (这部分是复用的)
        v = self.t_val_proj(Ft)  # (B, nt, H)
        v = v.view(batch_size, num_token, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, nt, hd)

        # 2. 应用 softmax 得到注意力权重 (这里使用缓存的 scores)
        # 注意：如果 cached_scores 已经包含了 mask 处理，这步可能不需要再次 mask？
        # 但为了保险起见，可以在外部确保 cached_scores 已处理，或在这里再处理一次。
        # 这里假设 cached_scores 已经经过了 mask 处理。
        attn_weights = F.softmax(cached_scores, dim=-1)  # (B, nh, vl, nt)
        attn_weights = self.attn_dropout(attn_weights)

        # 3. 加权聚合文本信息
        weighted_v = torch.matmul(attn_weights, v)  # (B, nh, vl, hd)

        # 4. 合并多头结果
        weighted_v = weighted_v.transpose(1, 2).contiguous()  # (B, vl, nh, hd)
        weighted_v = weighted_v.view(batch_size, vid_len, self.hidden_dim)  # (B, vl, H)

        # 5. 应用输出投影
        attn_output = self.out_proj(weighted_v)  # (B, vl, v_dim)
        attn_output = self.proj_dropout(attn_output)

        # 6. 残差连接和层归一化
        attn_output = self.norm(Fv + attn_output) # Residual from Fv

        # 7. 前馈网络
        ff_output = self.feed_forward(attn_output)

        # 8. 残差连接和最终层归一化
        Fv_enhanced = self.final_norm(attn_output + ff_output)

        # 9. 应用视频掩码（如果提供）
        if vid_mask is not None:
            mask_expanded = vid_mask.unsqueeze(-1).float()  # (B, vl, 1)
            Fv_enhanced = Fv_enhanced * mask_expanded

        return Fv_enhanced

        
    def forward(self, Ft, Fv, vid_mask=None, text_mask=None):
        """
        前向传播
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
            Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
            vid_mask: 视频掩码，形状为 (batch_size, vid_len)，True/1表示有效位置
            text_mask: 文本掩码，形状为 (batch_size, num_token)，True/1表示有效位置
            
        返回:
            Fv_enhanced: 增强后的视频特征，形状为 (batch_size, vid_len, v_dim)
        """
        batch_size, vid_len, _ = Fv.size()
        _, num_token, _ = Ft.size()
        
        # 1. 投影query, key, value
        q = self.v_proj(Fv)           # (batch_size, vid_len, hidden_dim)
        k = self.t_key_proj(Ft)       # (batch_size, num_token, hidden_dim)
        v = self.t_val_proj(Ft)       # (batch_size, num_token, hidden_dim)
        
        # 2. 分割多头
        head_dim = self.hidden_dim // self.num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, head_dim)  
        k = k.view(batch_size, -1, self.num_heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, head_dim)
        
        # 交换维度以便于计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, num_token, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, num_token, head_dim)
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 应用文本掩码（如果提供）
        if text_mask is not None:
            # 扩展文本掩码以匹配注意力分数的形状
            # (batch_size, num_token) -> (batch_size, 1, 1, num_token)
            expanded_text_mask = text_mask.unsqueeze(1).unsqueeze(2)
            
            # 在不有效的文本位置应用大的负值
            scores = scores.masked_fill(~expanded_text_mask.bool(), -1e9)
        
        # 4. 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, vid_len, num_token)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 5. 加权聚合文本信息
        weighted_v = torch.matmul(attn_weights, v)
        
        # 6. 合并多头结果
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        weighted_v = weighted_v.view(batch_size, vid_len, self.hidden_dim)
        
        # 7. 应用输出投影
        attn_output = self.out_proj(weighted_v)
        attn_output = self.proj_dropout(attn_output)
        
        # 8. 残差连接和层归一化
        attn_output = self.norm(Fv + attn_output)
        
        # 9. 前馈网络
        ff_output = self.feed_forward(attn_output)
        
        # 10. 残差连接和最终层归一化
        Fv_enhanced = self.final_norm(attn_output + ff_output)
        
        # 11. 应用视频掩码（如果提供）
        if vid_mask is not None:
            # 扩展掩码维度以匹配输出形状
            mask_expanded = vid_mask.unsqueeze(-1).float()  # (batch_size, vid_len, 1)
            Fv_enhanced = Fv_enhanced * mask_expanded
        return Fv_enhanced

class PresentGuidedVideoAttention(nn.Module):
    """
    基于视觉引导的视频特征注意力模块
    
    输入:
        Fg: 作为指导的视频特征，形状为 (batch_size, guide_vid_len, dim)
        Fv: 视频特征，形状为 (batch_size, vid_len, dim)
        vid_mask: Fv的视频掩码，形状为 (batch_size, vid_len)，True/1表示有效位置
        
    输出:
        Fv_enhanced: 增强后的视频特征，保持原始形状 (batch_size, vid_len, dim)
    """
    def __init__(self, dim, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        初始化函数
        
        参数:
            dim: 输入特征维度
            hidden_dim: 内部处理的隐藏维度
            num_heads: 多头注意力的头数
            dropout: 随机失活率
        """
        super(PresentGuidedVideoAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "隐藏维度必须能被头数整除"
        
        # 投影层 - 视频特征作为query，指导特征作为key和value
        self.v_proj = nn.Linear(dim, hidden_dim)
        self.g_key_proj = nn.Linear(dim, hidden_dim)
        self.g_val_proj = nn.Linear(dim, hidden_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(hidden_dim, dim)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, Fg, Fv, vid_mask=None):
        """
        前向传播
        
        参数:
            Fg: 指导特征，形状为 (batch_size, guide_vid_len, dim)
            Fv: 视频特征，形状为 (batch_size, vid_len, dim)
            vid_mask: 视频掩码，形状为 (batch_size, vid_len)，True/1表示有效位置
            
        返回:
            Fv_enhanced: 增强后的视频特征，形状为 (batch_size, vid_len, dim)
        """
        batch_size, vid_len, _ = Fv.size()
        _, guide_vid_len, _ = Fg.size()
        
        # 1. 投影query, key, value
        q = self.v_proj(Fv)              # (batch_size, vid_len, hidden_dim)
        k = self.g_key_proj(Fg)          # (batch_size, guide_vid_len, hidden_dim)
        v = self.g_val_proj(Fg)          # (batch_size, guide_vid_len, hidden_dim)
        
        # 2. 分割多头
        head_dim = self.hidden_dim // self.num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, head_dim)  
        k = k.view(batch_size, -1, self.num_heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, head_dim)
        
        # 交换维度以便于计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, guide_vid_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, guide_vid_len, head_dim)
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 4. 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, vid_len, guide_vid_len)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 5. 加权聚合指导特征信息
        weighted_v = torch.matmul(attn_weights, v)
        
        # 6. 合并多头结果
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        weighted_v = weighted_v.view(batch_size, vid_len, self.hidden_dim)
        
        # 7. 应用输出投影
        attn_output = self.out_proj(weighted_v)
        attn_output = self.proj_dropout(attn_output)
        
        # 8. 残差连接和层归一化
        attn_output = self.norm(Fv + attn_output)
        
        # 9. 前馈网络
        ff_output = self.feed_forward(attn_output)
        
        # 10. 残差连接和最终层归一化
        Fv_enhanced = self.final_norm(attn_output + ff_output)
        
        # 11. 应用视频掩码（如果提供）
        if vid_mask is not None:
            # 扩展掩码维度以匹配输出形状
            mask_expanded = vid_mask.unsqueeze(-1).float()  # (batch_size, vid_len, 1)
            Fv_enhanced = Fv_enhanced * mask_expanded
        
        return Fv_enhanced

class SimpleCompressor(nn.Module):
    """
    基于注意力机制的视频记忆压缩模块
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        mask: 指示有效位置的掩码，形状为 (batch_size, vid_len)。这个掩码同时应用于Fv和extern_weight
        
    输出:
        Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
    """

    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1):
        """
        初始化压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
        """
        super(SimpleCompressor, self).__init__()

        self.dimension = dimension
        self.compress_len = compress_len
        self.num_heads = num_heads
        
        # 确保dimension可以被num_heads整除
        assert dimension % num_heads == 0, "维度必须能被头数整除"

        # 可学习的查询参数，将作为cross-attention中的query
        # 形状: (1, compress_len, dimension)
        self.query = nn.Parameter(torch.randn(1, compress_len, dimension))
        
        # 初始化查询参数
        nn.init.xavier_uniform_(self.query)
        
        # 多头注意力层 

        # 为query, key, value创建线性变换
        self.q_proj = nn.Linear(dimension, dimension)
        self.k_proj = nn.Linear(dimension, dimension)
        self.v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # 层归一化和残差连接
        self.norm = nn.LayerNorm(dimension)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(dropout)
        )
        
        # 最终的层归一化
        self.final_norm = nn.LayerNorm(dimension)

    def forward(self, Fv, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
        """
        
        batch_size, vid_len, _ = Fv.size()
        
        # 扩展查询以匹配批次大小
        # 从 (1, compress_len, dimension) 到 (batch_size, compress_len, dimension)
        query = self.query.expand(batch_size, -1, -1)

        # 1. 投影query, key, value
        q = self.q_proj(query)  # (batch_size, compress_len, dimension)
        k = self.k_proj(Fv)     # (batch_size, vid_len, dimension)
        v = self.v_proj(Fv)     # (batch_size, vid_len, dimension)
        
        # 2. 分割多头
        head_dim = self.dimension // self.num_heads
        
        # 重塑为多头格式
        # (batch_size, seq_len, dimension) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, -1, self.num_heads, head_dim)
        k = k.view(batch_size, -1, self.num_heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, head_dim)
        
        # 交换维度以便于计算
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, compress_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        
        # 3. 计算注意力分数
        # (batch_size, num_heads, compress_len, head_dim) @ (batch_size, num_heads, head_dim, vid_len)
        # -> (batch_size, num_heads, compress_len, vid_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码以适应多头注意力的形状
            # (batch_size, vid_len) -> (batch_size, 1, 1, vid_len)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            
            # 在不有效的位置应用大的负值
            scores = scores.masked_fill(~expanded_mask.bool(), -1e9)
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, compress_len, vid_len)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 4. 原始注意力权重
        weighted_v = torch.matmul(attn_weights, v)
            
        # 5. 合并多头结果
        # (batch_size, num_heads, compress_len, head_dim) -> (batch_size, compress_len, num_heads, head_dim)
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        
        # (batch_size, compress_len, num_heads, head_dim) -> (batch_size, compress_len, dimension)
        weighted_v = weighted_v.view(batch_size, self.compress_len, self.dimension)

        # 6. 应用输出投影
        attn_output = self.out_proj(weighted_v)
        
        # 7. 残差连接和层归一化
        attn_output = self.norm(query + attn_output)
        
        # 8. 前馈网络
        ff_output = self.feed_forward(attn_output)
        
        # 9. 残差连接和最终层归一化
        Fv_compress = self.final_norm(attn_output + ff_output)

        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((batch_size, self.compress_len), dtype=torch.bool, device=Fv.device)
        
        return Fv_compress, compress_mask


class IntraMemoryStateEncoder(nn.Module):
    """
    Intra-memory 状态编码器：实现 Step1–3 的预处理逻辑。
    
    功能:
        1. 基于历史特征构造静态 + 动态拼接状态表示。
        2. 使用文本特征对历史状态做跨模态语义增强。
        3. 可选地根据显著性与时间距离构造先验偏置。
    """

    def __init__(
        self,
        feature_dim: int,
        saliency_alpha: float = 1.0,
        recency_gamma: float = 1.0,
        eps: float = 1e-6,
        use_mirror_semantic: bool = True,
        use_saliency_bias: bool = True,
        use_distance_bias: bool = True,
    ):
        """
        初始化状态编码器。
        
        输入:
            feature_dim: 单帧视频特征维度。
            saliency_alpha: 显著性先验缩放系数 α。
            recency_gamma: 距离先验缩放系数 γ。
            eps: 防止对 0 取对数的稳定项。
            use_saliency_bias: 是否启用显著性 bias。
            use_distance_bias: 是否启用距离 bias。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.saliency_alpha = saliency_alpha
        self.recency_gamma = recency_gamma
        self.eps = eps
        self.use_saliency_bias = use_saliency_bias
        self.use_distance_bias = use_distance_bias

        env_flag = os.getenv("LH_USE_MIRROR_SEMANTIC")
        if env_flag is not None:
            use_mirror_semantic = env_flag != "0"
        self.use_mirror_semantic = use_mirror_semantic

        self.state_norm = nn.LayerNorm(2 * feature_dim)

        self.semantic_enhancer = TextGuidedVideoAttention(
            t_dim=feature_dim,
            v_dim=2 * feature_dim,
            hidden_dim=2 * feature_dim,
        )

        self.output_proj = nn.Linear(2 * feature_dim, feature_dim)

    def build_temporal_feature(self, history_feat: torch.Tensor) -> torch.Tensor:
        """
        构造时间增强后的状态特征。
        
        输入:
            history_feat: 历史视频特征，形状为 [B, L, d]。
        输出:
            fused_state: 拼接并归一化后的特征，形状为 [B, L, 2d]。
        """
        delta_feat = torch.zeros_like(history_feat)
        delta_feat[:, 1:] = history_feat[:, 1:] - history_feat[:, :-1]

        fused_state = torch.cat([history_feat, delta_feat], dim=-1)
        fused_state = self.state_norm(fused_state)

        return fused_state

    def build_prior_bias(
        self,
        prev_saliency: Optional[torch.Tensor],
        frame_index: torch.Tensor,
        center_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于历史显著性与时间距离构造先验偏置。
        
        输入:
            prev_saliency: 历史显著性分数，形状为 [B, L] 或 None。
            frame_index: 每帧的全局下标 t_j，形状为 [B, L]。
            center_index: 当前片段中心下标 m，形状为 [B] 或 [B, 1]。
        输出:
            total_bias: 组合后的先验偏置，形状为 [B, L]。
        """
        if (not self.use_saliency_bias) or (prev_saliency is None):
            saliency_bias = 0.0
        else:
            safe_score = torch.clamp(prev_saliency, min=0.0)
            saliency_bias = self.saliency_alpha * torch.log(safe_score + self.eps)

        if self.use_distance_bias:
            center_index = center_index.view(-1, 1).float()
            time_distance = torch.abs(frame_index.float() - center_index)
            distance_bias = self.recency_gamma * torch.exp(-time_distance)
        else:
            distance_bias = 0.0

        total_bias = saliency_bias + distance_bias

        # 验证阶段常无 prev_saliency；若两项 bias 均为标量 0，相加结果为 float，
        # 下游 IGEC 的 apply_mask_and_bias 需要 [B,L] 张量。
        if not isinstance(total_bias, torch.Tensor):
            try:
                total_bias = torch.zeros(
                    frame_index.shape,
                    dtype=torch.float32,
                    device=frame_index.device,
                )
            except Exception as exc:
                raise RuntimeError(
                    "无法将先验偏置规范为与 frame_index 同设备的张量。"
                ) from exc

        return total_bias

    def forward(
        self,
        history_feat: torch.Tensor,
        text_feat: torch.Tensor,
        vid_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        prev_saliency: Optional[torch.Tensor] = None,
        frame_index: Optional[torch.Tensor] = None,
        center_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向过程：生成状态特征、原始 2d 状态以及可选先验偏置。
        
        输出:
            encoded_state: 编码后的状态特征，形状为 [B, L, d]。
            enhanced_state: 文本语义增强后的 2d 状态，[B, L, 2d]。
            prior_bias: 组合先验偏置，[B, L] 或 None。
        """
        fused_state = self.build_temporal_feature(history_feat)

        if self.use_mirror_semantic:
            enhanced_state = self.semantic_enhancer(
                Ft=text_feat,
                Fv=fused_state,
                vid_mask=vid_mask,
                text_mask=text_mask,
            )
        else:
            enhanced_state = fused_state

        encoded_state = self.output_proj(enhanced_state)

        if frame_index is None or center_index is None:
            prior_bias = None
        else:
            prior_bias = self.build_prior_bias(
                prev_saliency=prev_saliency,
                frame_index=frame_index,
                center_index=center_index,
            )

        return encoded_state, enhanced_state, prior_bias


class IntraMemoryIGEC(nn.Module):
    """
    Intra-memory 语义引导结构化压缩模块（Step4）。
    
    功能:
        1. 维护一组可学习状态锚点 M_c。
        2. 基于文本全局语义构造联合查询 Q_joint。
        3. 使用带先验偏置的注意力，从 F_state_2d 中采样结构化记忆。
    """

    def __init__(
        self,
        feature_dim: int,
        num_anchors: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        初始化 IGEC 模块。
        
        输入:
            feature_dim: 单通道特征维度 d。
            num_anchors: 状态锚点数量 T_c。
            num_heads: 多头注意力头数。
            dropout: 注意力权重的 Dropout 比例。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = feature_dim * 2
        self.num_anchors = num_anchors
        self.num_heads = num_heads

        assert self.hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.state_anchors = nn.Parameter(
            torch.randn(num_anchors, self.hidden_dim)
        )

        self.anchor_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.text_query_proj = nn.Linear(feature_dim, self.hidden_dim)

        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def build_joint_query(
        self,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造联合查询 Q_joint。
        
        输入:
            text_feat: 文本特征，[B, L_txt, d]。
        输出:
            q_joint: 联合查询，[B, T_c, 2d]。
        """
        batch_size = text_feat.size(0)

        text_summary = text_feat.mean(dim=1)
        text_query = self.text_query_proj(text_summary)
        text_query = text_query.unsqueeze(1).expand(
            batch_size, self.num_anchors, -1
        )

        anchors = self.state_anchors.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        anchor_query = self.anchor_query_proj(anchors)

        q_joint = anchor_query + text_query

        return q_joint

    def project_kv(
        self,
        state_feat_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 2d 状态特征投影为注意力的 K、V。
        
        输入:
            state_feat_2d: 状态特征，[B, L, 2d]。
        输出:
            key_proj: K，[B, L, 2d]。
            value_proj: V，[B, L, 2d]。
        """
        key_proj = self.key_proj(state_feat_2d)
        value_proj = self.value_proj(state_feat_2d)

        return key_proj, value_proj

    def apply_mask_and_bias(
        self,
        scores: torch.Tensor,
        vid_mask: Optional[torch.Tensor],
        prior_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        对注意力打分应用视频掩码与先验偏置。
        
        输入:
            scores: 原始打分，[B, H, T_c, L]。
            vid_mask: 视频掩码，[B, L] 或 None。
            prior_bias: 先验偏置，[B, L] 或 None。
        输出:
            scores: 加工后的打分,[B, H, T_c, L]。
        """
        if vid_mask is not None:
            mask_bool = vid_mask.bool()
            mask = mask_bool.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e9)

        if prior_bias is not None:
            bias = prior_bias.unsqueeze(1).unsqueeze(2)
            scores = scores + bias

        return scores

    def forward(
        self,
        state_feat_2d: torch.Tensor,
        text_feat: torch.Tensor,
        prior_bias: Optional[torch.Tensor] = None,
        vid_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：执行 IGEC，得到结构化记忆表示。
        
        输入:
            state_feat_2d: 文本增强后的状态特征，[B, L, 2d]。
            text_feat: 文本特征，[B, L_txt, d]。
            prior_bias: 先验偏置，[B, L] 或 None。
            vid_mask: 视频掩码，[B, L] 或 None。
            text_mask: 文本掩码（暂未使用），保留以便后续扩展。
        输出:
            m_raw: 采样后的结构化记忆，[B, T_c, 2d]。
            attn_weights: 注意力权重，[B, H, T_c, L]。
        """
        del text_mask

        batch_size, seq_len, _ = state_feat_2d.shape
        head_dim = self.hidden_dim // self.num_heads

        q_joint = self.build_joint_query(text_feat)
        key_proj, value_proj = self.project_kv(state_feat_2d)

        q = q_joint.view(batch_size, self.num_anchors, self.num_heads, head_dim)
        q = q.transpose(1, 2)

        k = key_proj.view(batch_size, seq_len, self.num_heads, head_dim)
        k = k.transpose(1, 2)

        v = value_proj.view(batch_size, seq_len, self.num_heads, head_dim)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = self.apply_mask_and_bias(scores, vid_mask, prior_bias)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, self.num_anchors, self.hidden_dim)

        return context, attn_weights

class TextGuidedCompressor(nn.Module):
    """
    基于文本引导的视频记忆压缩模块，使用门控机制融合两个分支的结果
    """
    def __init__(
        self,
        t_dim,
        v_dim,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        compress_len: int = 10,
        weight_alpha=None,
        intra_output_mode: str = "fuse",
        compressor_output_mode: str = "legacy_fuse",
        ablate_delta_stream: bool = False,
        use_saliency_bias: bool = True,
        use_distance_bias: bool = True,
    ):
        """
        初始化文本引导的视频压缩器。
        
        参数:
            t_dim: 文本特征维度。
            v_dim: 视频特征维度。
            hidden_dim: 注意力机制内部隐藏维度。
            num_heads: 多头注意力的头数。
            dropout: Dropout 率。
            compress_len: 压缩后的序列长度。
            weight_alpha: 外部权重缩放系数。
            intra_output_mode: Intra memory 输出模式，取值为
                "fuse"（默认，静态与动态通道门控融合）、
                "static"（仅输出静态通道）、
                "delta"（仅输出动态通道）。
            compressor_output_mode: 压缩器最终输出模式，取值为
                "legacy_fuse"（默认，保持当前 direct + intra 的融合输出）、
                "dual_intra"（输出 intra 的双路分支，便于后续 inter-memory 对接）。
            ablate_delta_stream: 是否将 delta 分支替换为 static 分支输出。
            use_saliency_bias: 是否启用显著性 bias。
            use_distance_bias: 是否启用距离 bias。
        """
        super().__init__()

        self.compress_len = compress_len
        self.ablate_delta_stream = ablate_delta_stream

        env_mode = os.getenv("LH_INTRA_OUTPUT_MODE")
        if env_mode is not None:
            intra_output_mode = env_mode
        self.intra_output_mode = intra_output_mode

        output_mode_env = os.getenv("LH_COMPRESSOR_OUTPUT_MODE")
        if output_mode_env is not None:
            compressor_output_mode = output_mode_env
        self.compressor_output_mode = compressor_output_mode

        # 状态编码器：实现 Step1–3 的历史特征预处理
        self.state_encoder = IntraMemoryStateEncoder(
            feature_dim=v_dim,
            saliency_alpha=1.0,
            recency_gamma=1.0,
            use_saliency_bias=use_saliency_bias,
            use_distance_bias=use_distance_bias,
        )

        # IGEC：实现 Step4 的语义引导结构化压缩
        self.use_igec = True
        self.igec = IntraMemoryIGEC(
            feature_dim=v_dim,
            num_anchors=compress_len,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 旧版本的文本引导注意力暂时保留，后续如需兼容可继续复用
        self.text_guided_video_attention = TextGuidedVideoAttention(
            t_dim=t_dim,
            v_dim=v_dim,
            hidden_dim=hidden_dim)
            
        # 文本增强视频特征压缩器
        self.text_enhanced_memory_compressor = SimpleCompressor(
            v_dim, 
            compress_len=compress_len)
        if weight_alpha is None:
            # 文本增强视频特征压缩器
            self.direct_memory_compressor = SimpleCompressor(
                v_dim, 
                compress_len=compress_len)
        else:
            self.direct_memory_compressor = CompressorWithExternalWeights(
                v_dim, 
                compress_len=compress_len,
                weight_alpha=weight_alpha)
        # 直接特征门控 - 控制原始视频特征的贡献
        self.direct_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim<<1),  # v_dim -> hidden_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),      # hidden_dim*2 -> 1
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # 文本增强特征门控 - 控制文本增强视频特征的贡献
        self.text_enhanced_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim<<1),  # v_dim -> hidden_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),      # hidden_dim*2 -> 1
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        # # 新增：逐 token 门控（无偏置，省参数）
        # self.token_gate = nn.Sequential(
        #     nn.Linear(v_dim * 2, 1, bias=False),  # 拼接 → 1
        #     nn.Sigmoid()                          # 0~1，可直接当权重
        # )
        
        # 添加层归一化
        self.norm = nn.LayerNorm(v_dim, elementwise_affine=False)

        # 缓存最近一次前向过程中的先验偏置，后续 Step4/5 可以直接复用
        self.last_prior_bias: Optional[torch.Tensor] = None

        # Step5 中使用的最近帧门控网络，输出静态与动态通道的权重
        self.recent_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim << 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim << 1, 2),
        )

    def build_recent_summary(
        self,
        Fv: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造最近帧的摘要特征。
        
        输入:
            Fv: 历史视频特征，[B, L, d]。
        输出:
            summary: 最近若干帧的平均特征，[B, d]。
        """
        batch_size, seq_len, feat_dim = Fv.shape
        _ = batch_size
        recent_len = min(self.compress_len, seq_len)

        recent_feat = Fv[:, -recent_len:, :].reshape(
            batch_size, recent_len, feat_dim
        )

        summary = recent_feat.mean(dim=1)

        return summary

    def fuse_static_and_delta(
        self,
        m_raw: torch.Tensor,
        recent_summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step5: 通道分离与基于最近帧的门控融合。
        
        输入:
            m_raw: IGEC 输出的结构化记忆，[B, T_c, 2d]。
            recent_summary: 最近帧摘要特征，[B, d]。
        输出:
            m_intra: 融合后的 intra memory，[B, T_c, d]，或单一通道。
        """
        batch_size, num_tokens, two_d = m_raw.shape
        _ = num_tokens
        d = two_d // 2

        m_static = m_raw[:, :, :d]
        m_delta = m_raw[:, :, d:]

        mode = self.intra_output_mode
        if mode == "static":
            return m_static
        if mode == "delta":
            return m_delta

        gate_logits = self.recent_gate(recent_summary)
        gate = torch.sigmoid(gate_logits)

        g_static = gate[:, 0].view(batch_size, 1, 1)
        g_delta = gate[:, 1].view(batch_size, 1, 1)

        m_intra = g_static * m_static + g_delta * m_delta

        return m_intra

    def split_static_and_delta(
        self,
        m_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 IGEC 输出拆分为静态分支与动态分支。
        
        输入:
            m_raw: IGEC 输出的结构化记忆，[B, T_c, 2d]。
        输出:
            m_static: 静态语义分支，[B, T_c, d]。
            m_delta: 动态位移分支，[B, T_c, d]。
        """
        _, _, two_d = m_raw.shape
        d = two_d // 2
        m_static = m_raw[:, :, :d]
        m_delta = m_raw[:, :, d:]

        return m_static, m_delta

    def forward(
        self,
        Ft,
        Fv,
        vid_mask=None,
        text_mask=None,
        extern_weight=None,
        prev_saliency: Optional[torch.Tensor] = None,
        frame_index: Optional[torch.Tensor] = None,
        center_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（集成 Step1–3 的新版实现）。
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)。
            Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)。
            vid_mask: 视频掩码，形状为 (batch_size, vid_len)。
            text_mask: 文本掩码，形状为 (batch_size, num_token)。
            extern_weight: 外部显著性权重，形状为 (batch_size, vid_len) 或 None。
            prev_saliency: 历史显著性分数，形状为 (batch_size, vid_len) 或 None。
            frame_index: 每帧的全局下标，形状为 (batch_size, vid_len) 或 None。
            center_index: 当前片段中心下标，形状为 (batch_size,) 或 None。
            
        返回:
            legacy_fuse 模式:
                feat: 融合后的压缩视频特征，形状为 (batch_size, compress_len, v_dim)。
                combined_weights: 文本增强分支的注意力权重。
            dual_intra 模式:
                feat: 一个字典，包含 static/delta/fused 三路 intra 特征。
                combined_weights: IGEC 注意力权重。
        """
        valid_output_modes = {"legacy_fuse", "dual_intra"}
        if self.compressor_output_mode not in valid_output_modes:
            raise ValueError(
                "compressor_output_mode 必须是 'legacy_fuse' 或 'dual_intra'。"
            )

        encoded_state, state_2d, prior_bias = self.state_encoder(
            history_feat=Fv,
            text_feat=Ft,
            vid_mask=vid_mask,
            text_mask=text_mask,
            prev_saliency=prev_saliency,
            frame_index=frame_index,
            center_index=center_index,
        )

        self.last_prior_bias = prior_bias
        if self.use_igec:
            m_raw, combined_weights = self.igec(
                state_feat_2d=state_2d,
                text_feat=Ft,
                prior_bias=prior_bias,
                vid_mask=vid_mask,
                text_mask=text_mask,
            )

            m_static, m_delta = self.split_static_and_delta(m_raw)
            if self.ablate_delta_stream:
                m_delta = m_static
                m_raw = torch.cat([m_static, m_delta], dim=-1)
            recent_summary = self.build_recent_summary(Fv)
            Fv_compress_enhanced = self.fuse_static_and_delta(
                m_raw=m_raw,
                recent_summary=recent_summary,
            )
        else:
            Fv_compress_enhanced, combined_weights = (
                self.text_enhanced_memory_compressor(
                    encoded_state,
                    mask=vid_mask,
                )
            )
            m_static = Fv_compress_enhanced
            m_delta = torch.zeros_like(Fv_compress_enhanced)

        if self.compressor_output_mode == "dual_intra":
            dual_intra_feat: Dict[str, torch.Tensor] = {
                "static": m_static,
                "delta": m_delta,
                "fused": Fv_compress_enhanced,
            }
            return dual_intra_feat, combined_weights

        if extern_weight is None:
            Fv_compress, _ = self.direct_memory_compressor(Fv, mask=vid_mask)
        else:
            Fv_compress, _ = self.direct_memory_compressor(
                Fv,
                mask=vid_mask,
                extern_weight=extern_weight,
            )

        direct_gate_value = self.direct_gate(Fv_compress.mean(1, keepdim=True))
        text_enhanced_gate_value = self.text_enhanced_gate(
            Fv_compress_enhanced.mean(1, keepdim=True)
        )

        feat = self.norm(
            direct_gate_value * Fv_compress
            + text_enhanced_gate_value * Fv_compress_enhanced
        )

        return feat, combined_weights

class TextPresentGuidedCompressor(nn.Module):
    """
    基于文本和视觉引导的视频记忆压缩模块，使用门控机制融合两个分支的结果
    
    两个分支:
    1. 文本引导分支: TextGuidedVideoAttention + SimpleCompressor
    2. 视觉引导分支: PresentGuidedVideoAttention + SimpleCompressor
    
    输入:
        Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
        Fg: 视觉引导特征，形状为 (batch_size, guide_vid_len, v_dim)
        Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
        vid_mask: 视频掩码，形状为 (batch_size, vid_len)
        text_mask: 文本掩码，形状为 (batch_size, num_token)
        
    输出:
        feat: 融合后的压缩视频特征，形状为 (batch_size, compress_len, v_dim)
        mask: 压缩后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(self, t_dim, v_dim, hidden_dim=512, num_heads=8, dropout=0.1, compress_len=10):
        """
        初始化文本和视觉引导的视频压缩器
        
        参数:
            t_dim: 文本特征维度
            v_dim: 视频特征维度
            hidden_dim: 注意力机制内部隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            compress_len: 压缩后的序列长度
        """
        super().__init__()

        self.compress_len = compress_len

        # 文本引导视频注意力模块
        self.text_guided_video_attention = TextGuidedVideoAttention(
            t_dim=t_dim,
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 视觉引导视频注意力模块
        self.present_guided_video_attention = PresentGuidedVideoAttention(
            dim=v_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
            
        # 文本增强视频特征压缩器
        self.text_enhanced_memory_compressor = SimpleCompressor(
            v_dim, 
            compress_len=compress_len,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 视觉增强视频特征压缩器
        self.present_enhanced_memory_compressor = SimpleCompressor(
            v_dim, 
            compress_len=compress_len,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 文本增强特征门控 - 控制文本增强视频特征的贡献
        self.text_enhanced_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim<<1),  # v_dim -> hidden_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),      # hidden_dim*2 -> 1
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # 视觉增强特征门控 - 控制视觉增强视频特征的贡献
        self.present_enhanced_gate = nn.Sequential(
            nn.Linear(v_dim, hidden_dim<<1),  # v_dim -> hidden_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),      # hidden_dim*2 -> 1
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # 添加层归一化
        self.norm = nn.LayerNorm(v_dim, elementwise_affine=False)

    def forward(self, Ft, Fg, Fv, vid_mask=None, text_mask=None):
        """
        前向传播
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
            Fg: 视觉引导特征，形状为 (batch_size, guide_vid_len, v_dim)
            Fv: 视频特征，形状为 (batch_size, vid_len, v_dim)
            vid_mask: 视频掩码，形状为 (batch_size, vid_len)
            text_mask: 文本掩码，形状为 (batch_size, num_token)
            
        返回:
            feat: 融合后的压缩视频特征，形状为 (batch_size, compress_len, v_dim)
            mask: 压缩后的掩码，形状为 (batch_size, compress_len)
        """
        # 文本引导增强的视频特征 - (batch_size, vid_len, v_dim)
        Fv_enhanced_text = self.text_guided_video_attention(Ft, Fv, vid_mask, text_mask)
        
        # 视觉引导增强的视频特征 - (batch_size, vid_len, v_dim)
        Fv_enhanced_present = self.present_guided_video_attention(Fg, Fv, vid_mask)
        
        # 压缩文本增强的视频特征 - (batch_size, compress_len, v_dim)
        Fv_compress_text, mask_text = self.text_enhanced_memory_compressor(Fv_enhanced_text, mask=vid_mask)
        
        # 压缩视觉增强的视频特征 - (batch_size, compress_len, v_dim)
        Fv_compress_present, mask_present = self.present_enhanced_memory_compressor(Fv_enhanced_present, mask=vid_mask)

        # 计算门控值
        # 文本增强特征的门控值 - (batch_size, 1, 1)
        text_enhanced_gate_value = self.text_enhanced_gate(Fv_compress_text.mean(1, keepdim=True))
        
        # 视觉增强特征的门控值 - (batch_size, 1, 1)
        present_enhanced_gate_value = self.present_enhanced_gate(Fv_compress_present.mean(1, keepdim=True))

        # --- 融合两个分支特征 ---
        # text_enhanced_gate_value * Fv_compress_text - (batch_size, compress_len, v_dim)
        # present_enhanced_gate_value * Fv_compress_present - (batch_size, compress_len, v_dim)
        feat = self.norm(
            text_enhanced_gate_value * Fv_compress_text + 
            present_enhanced_gate_value * Fv_compress_present
        )

        return feat, mask_text

class MultimodalTokenCompressor(nn.Module):
    def __init__(self, v_dim, compress_len, num_heads=8, dropout=0.1):
        super().__init__()
        self.compress_len = compress_len
        self.token_learner = MultimodalTokenLearner(
            hidden_dim=v_dim,
            num_tokens=compress_len,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, Fv, query=None, mask=None, query_mask=None):
        """
        Args:
            Fv: (bsz, v_len, v_dim)
            query: (bsz, q_len, v_dim)
            mask: [B, T]
        Returns:
            compressed_feats: (bsz, compress_len, v_dim)
            weights: (bsz, compress_len, v_len)
        """
        # if query is None:
        #     # 若无查询，用均值作为伪查询
        #     query = video_feats.mean(dim=1, keepdim=True)  # (bsz, 1, v_dim)
        assert query is not None, "Query must be provided for multimodal compression"
        
        compressed_feats = self.token_learner(Fv, query, mask, query_mask)
        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((Fv.shape[0], self.compress_len), dtype=torch.bool, device=Fv.device)
        return compressed_feats, compress_mask

class MultimodalTokenLearner(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.build_tokenlearner(hidden_dim, num_tokens, dropout)

    def build_tokenlearner(self, hidden_dim, num_tokens, dropout):
        # Vision tokenlearner
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout)
        )
        self.video_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim<<1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        # Linguistic tokenlearner
        self.fc_feat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.w = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, num_tokens, bias=False)
        )
        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim<<1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)


    def tokenlearner(self, input, query, input_mask=None, query_mask=None):
        """Args:
            input:  [B, T, h] 视频特征
            query:  [B, q_len, h] 查询特征
            input_mask: [B, T]
            query_mask: [B, q_len]
        """
        # --- 视觉TokenLearner分支 ---
        selected = self.input_mlp(input)  # [B, T, n]
        selected = selected.transpose(1, 2)  # [B, n, T]
        
        # 应用掩码：将无效位置的权重设为负无穷
        if input_mask is not None:
            input_mask = input_mask.unsqueeze(1)  # [B, 1, T]
            selected = selected.masked_fill(~input_mask.bool(), -1e9)
        
        selected = F.softmax(selected, dim=-1)  # [B, n, T]
        feat_selected = torch.einsum('...nt,...td->...nd', selected, input)  # [B, n, h]

        # --- 语言引导分支 ---
        if query_mask is not None:
            query = (query * query_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
                    query_mask.sum(dim=1, keepdim=True).clamp(min=1e-6).unsqueeze(-1)
        else:
            query = query.mean(dim=1, keepdim=True)  # [B, 1, h]
        attn = self.fc_feat(input) + self.fc_query(query)  # [B, T, h]
        attn = self.w(attn).transpose(1, 2)  # [B, n, T]
        
        # 应用掩码（同上）
        if input_mask is not None:
            attn = attn.masked_fill(~input_mask.bool(), -1e9)
        
        attn = F.softmax(attn, dim=-1)  # [B, n, T]
        feat_query = torch.einsum('...nt,...td->...nd', attn, input)  # [B, n, h]

        # --- 融合 ---
        feat = self.norm(
            self.video_gate(input.mean(1, keepdim=True)) * feat_selected + 
            self.query_gate(query) * feat_query
        )
        return feat

    def forward(self, input, query, mask=None, query_mask=None):
        return self.tokenlearner(input, query, mask, query_mask)

# 0. 基础的基于注意力机制的视频记忆压缩模块，支持融合外部权重
class CompressorWithExternalWeights(nn.Module):
    """
    基于注意力机制的视频记忆压缩模块，支持融合外部权重
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        mask: 指示有效位置的掩码，形状为 (batch_size, vid_len)。这个掩码同时应用于Fv和extern_weight
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        
    输出:
        Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
        mask: 压缩后的有效掩码，形状为 (batch_size, compress_len)
    """

    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5):
        """
        初始化压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
        """
        super(CompressorWithExternalWeights, self).__init__()

        self.dimension = dimension
        self.compress_len = compress_len
        self.num_heads = num_heads

        self.weight_alpha = weight_alpha
        
        # 确保dimension可以被num_heads整除
        assert dimension % num_heads == 0, "维度必须能被头数整除"

        # 可学习的查询参数，将作为cross-attention中的query
        # 形状: (1, compress_len, dimension)
        self.query = nn.Parameter(torch.randn(1, compress_len, dimension))
        
        # 初始化查询参数
        nn.init.xavier_uniform_(self.query)
        
        # 多头注意力层 - 不直接使用nn.MultiheadAttention，因为需要获取并修改注意力权重
        # 因此需要自己实现多头注意力机制的核心部分

        # 为query, key, value创建线性变换
        self.q_proj = nn.Linear(dimension, dimension)
        self.k_proj = nn.Linear(dimension, dimension)
        self.v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # 层归一化和残差连接
        self.norm = nn.LayerNorm(dimension)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(dropout)
        )
        
        # 最终的层归一化
        self.final_norm = nn.LayerNorm(dimension)

    def before_attn(self, Fv, mask=None):
        """
        在注意力机制之前进行操作
        """
        batch_size, vid_len, _ = Fv.size()
        
        # 扩展查询以匹配批次大小
        # 从 (1, compress_len, dimension) 到 (batch_size, compress_len, dimension)
        query = self.query.expand(batch_size, -1, -1)

        # 1. 投影query, key, value
        q = self.q_proj(query)  # (batch_size, compress_len, dimension)
        k = self.k_proj(Fv)     # (batch_size, vid_len, dimension)
        v = self.v_proj(Fv)     # (batch_size, vid_len, dimension)
        
        # 2. 分割多头
        head_dim = self.dimension // self.num_heads
        
        # 重塑为多头格式
        # (batch_size, seq_len, dimension) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, -1, self.num_heads, head_dim)
        k = k.view(batch_size, -1, self.num_heads, head_dim)
        v = v.view(batch_size, -1, self.num_heads, head_dim)
        
        # 交换维度以便于计算
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, compress_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, vid_len, head_dim)
        
        # 3. 计算注意力分数
        # (batch_size, num_heads, compress_len, head_dim) @ (batch_size, num_heads, head_dim, vid_len)
        # -> (batch_size, num_heads, compress_len, vid_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码以适应多头注意力的形状
            # (batch_size, vid_len) -> (batch_size, 1, 1, vid_len)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            
            # 在不有效的位置应用大的负值
            scores = scores.masked_fill(~expanded_mask.bool(), -1e9)
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, compress_len, vid_len)
        attn_weights = self.attn_dropout(attn_weights)
        return attn_weights, query, v
    
    def ex_weight_fusion(self, mask, batch_size, vid_len, attn_weights, extern_weight):
        # 4. 如果提供了外部权重，将其与注意力权重融合
        combined_weights = None
        if extern_weight is not None:
            # 确保extern_weight的形状正确
            assert extern_weight.shape == (batch_size, vid_len), \
                f"外部权重形状应为 {(batch_size, vid_len)}, 但得到了 {extern_weight.shape}"
            
            # 首先将extern_weight形状由(batch_size, vid_len)变为(batch_size, compress_len, vid_len)
            # 扩展后的维度1应该为compress_len，每个维度2的值相同
            extern_weight = extern_weight.unsqueeze(1).expand(-1, self.compress_len, -1)

            # 首先处理外部权重的掩码，同样使用传入的mask
            masked_extern_weight = extern_weight.clone()

            if mask is not None:
                # 将mask应用于extern_weight的vid_len维度
                # mask: (batch_size, vid_len)
                # extern_weight: (batch_size, compress_len, vid_len)
                mask_expanded = mask.unsqueeze(1)  # (batch_size, 1, vid_len)

                masked_extern_weight = masked_extern_weight.masked_fill(~mask_expanded.bool(), 0)
                
                # 重新归一化外部权重，使每行的和为1
                # 对于全为0的行（即所有位置都被掩码的情况），添加一个小的epsilon以避免除以0
                row_sums = masked_extern_weight.sum(dim=-1, keepdim=True)
                masked_extern_weight = masked_extern_weight / (row_sums + 1e-9)

            # 扩展外部权重以匹配多头注意力的形状
            # (batch_size, compress_len, vid_len) -> (batch_size, 1, compress_len, vid_len)
            extern_weight_expanded = masked_extern_weight.unsqueeze(1)

            # 融合两种权重
            combined_weights = self.weight_alpha * attn_weights + (1 - self.weight_alpha) * extern_weight_expanded
            
            # 确保权重总和为1
            # 由于已经对两个输入分别进行了softmax/归一化，并且是线性组合，
            # 所以combined_weights的每一行的和应该接近1，但为了确保精确性，我们再次归一化
            combined_weights_sum = combined_weights.sum(dim=-1, keepdim=True)
            combined_weights = combined_weights / (combined_weights_sum + 1e-9)
            
            # 使用融合后的权重计算加权和
            # (batch_size, num_heads, compress_len, vid_len) @ (batch_size, num_heads, vid_len, head_dim)
            # -> (batch_size, num_heads, compress_len, head_dim)
            # weighted_v = torch.matmul(combined_weights, v)
        else:
            # 如果没有外部权重，就使用原始注意力权重
            # weighted_v = torch.matmul(attn_weights, v)
            combined_weights = attn_weights
        return combined_weights

    def after_attn(self, weighted_v, query, batch_size, device):
        # 5. 合并多头结果
        # (batch_size, num_heads, compress_len, head_dim) -> (batch_size, compress_len, num_heads, head_dim)
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        
        # (batch_size, compress_len, num_heads, head_dim) -> (batch_size, compress_len, dimension)
        weighted_v = weighted_v.view(batch_size, self.compress_len, self.dimension)

        # 6. 应用输出投影
        attn_output = self.out_proj(weighted_v)
        
        # 7. 残差连接和层归一化
        attn_output = self.norm(query + attn_output)
        
        # 8. 前馈网络
        ff_output = self.feed_forward(attn_output)
        
        # 9. 残差连接和最终层归一化
        Fv_compress = self.final_norm(attn_output + ff_output)

        # 创建压缩后的掩码，形状为 (batch_size, compress_len)，全为True
        compress_mask = torch.ones((batch_size, self.compress_len), dtype=torch.bool, device=device)
        
        return Fv_compress, compress_mask
        

    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            combined_weights: 注意力权重与外部权重的平均，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, _ = Fv.size()

        attn_weights, query, v = self.before_attn(Fv, mask)
        
        combined_weights = self.ex_weight_fusion(mask, batch_size, vid_len, attn_weights, extern_weight)

        weighted_v = torch.matmul(combined_weights, v)

        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, Fv.device)
        
        return Fv_compress, compress_mask

# 0+. 加入位置权重的视频压缩模块，保留时序信息
class CompressorWithPositionWeights(CompressorWithExternalWeights):
    """
    基于注意力机制的视频记忆压缩模块，融合外部权重和位置权重
    
    位置权重的目的是在压缩时保留视频特征之间的时序关系，
    方法是为每个压缩后的特征位置设定"中心点"，
    基于原始特征与这些中心点的距离计算position_weight
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        mask: 指示有效位置的掩码，形状为 (batch_size, vid_len)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        
    输出:
        Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的有效掩码，形状为 (batch_size, compress_len)
    """

    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, 
                 attn_weight_alpha=0.33, extern_weight_alpha=0.33, position_weight_alpha=0.34):
        """
        初始化压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            attn_weight_alpha: 注意力权重的融合系数
            extern_weight_alpha: 外部权重的融合系数
            position_weight_alpha: 位置权重的融合系数
        """
        # 调用父类初始化，但不使用父类的weight_alpha
        super(CompressorWithPositionWeights, self).__init__(
            dimension, compress_len, num_heads, dropout, weight_alpha=None)
        
        # 存储三种权重的融合系数
        self.attn_weight_alpha = attn_weight_alpha
        self.extern_weight_alpha = extern_weight_alpha
        self.position_weight_alpha = position_weight_alpha
        
        # 检查权重和为1
        total_alpha = attn_weight_alpha + extern_weight_alpha + position_weight_alpha
        assert abs(total_alpha - 1.0) < 1e-5, f"权重和应为1，但得到了{total_alpha}"

    def _get_valid_lengths(self, mask):
        """
        计算每个样本的有效长度
        
        参数:
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            valid_lengths: 每个样本的有效长度，形状为 (batch_size,)
        """
        if mask is None:
            # 如果没有提供掩码，假设所有位置都有效
            return torch.full((mask.size(0),), mask.size(1), device=mask.device)
        else:
            # 计算每个样本的有效位置数量
            return mask.sum(dim=1).int()
    
    def _compute_position_weights(self, mask, batch_size, vid_len):
        """
        计算基于位置的权重
        
        参数:
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            batch_size: 批次大小
            vid_len: 视频长度
            
        返回:
            position_weights: 位置权重，形状为 (batch_size, compress_len, vid_len)
            need_compress: 指示每个样本是否需要压缩的掩码，形状为 (batch_size,)
            original_indices: 对于不需要压缩的样本，原始特征的索引，形状为 (batch_size, compress_len)
        """
        device = mask.device if mask is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算每个样本的有效长度
        valid_lengths = self._get_valid_lengths(mask)
        
        # 确定哪些样本需要压缩（有效长度大于compress_len）
        need_compress = valid_lengths > self.compress_len
        
        # 初始化位置权重和原始索引
        position_weights = torch.zeros(batch_size, self.compress_len, vid_len, device=device)
        original_indices = torch.zeros(batch_size, self.compress_len, dtype=torch.long, device=device)
        
        # 对每个样本计算位置权重
        for i in range(batch_size):
            if need_compress[i]:
                # 样本需要压缩
                valid_len = valid_lengths[i].item()
                
                # 在有效长度范围内找到有效位置的索引
                if mask is not None:
                    valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
                else:
                    valid_indices = torch.arange(valid_len, device=device)
                
                # 计算中心点（在有效特征中均匀分布）
                # 例如，如果valid_len=10, compress_len=4，中心点将在索引 1.25, 3.75, 6.25, 8.75
                # 取整后为 1, 4, 6, 9
                centers = torch.linspace(0, valid_len - 1, self.compress_len, device=device)
                centers = centers.round().long()
                
                # 从valid_indices中获取实际的中心点索引
                center_indices = valid_indices[centers]
                
                # 为每个中心点计算到每个有效位置的距离权重
                for j, center_idx in enumerate(center_indices):
                    # 计算中心点到每个位置的归一化距离
                    # 距离被归一化到[-1, 1]区间，中心点处为0
                    if mask is not None:
                        distances = torch.arange(vid_len, device=device)
                        distances = (distances - center_idx) / valid_len
                    else:
                        distances = torch.arange(vid_len, device=device)
                        distances = (distances - center_idx) / valid_len
                    
                    # 转换为权重 (1 - |distance|)，距离越近权重越大
                    weights = 1.0 - torch.abs(distances)
                    
                    # 确保只有有效位置有权重
                    if mask is not None:
                        weights = weights * mask[i].float()
                    
                    # 存储权重
                    position_weights[i, j] = weights
            else:
                # 样本不需要压缩，直接选择原始特征的前compress_len个
                valid_len = valid_lengths[i].item()
                
                # 找到有效位置的索引
                if mask is not None:
                    valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
                else:
                    valid_indices = torch.arange(vid_len, device=device)
                
                # 选择前compress_len个有效位置（或者所有有效位置）
                num_indices = min(valid_len, self.compress_len)
                selected_indices = valid_indices[:num_indices]
                
                # 将选定的索引存储起来
                original_indices[i, :num_indices] = selected_indices
                
                # 对选中的位置设置权重为1，其他位置为0
                for j, idx in enumerate(selected_indices):
                    position_weights[i, j, idx] = 1.0
        
        # 确保每行的权重和为1（对于有效权重）
        # 避免除以0，添加一个小的epsilon
        row_sums = position_weights.sum(dim=-1, keepdim=True)
        position_weights = position_weights / (row_sums + 1e-9)
        
        return position_weights, need_compress, original_indices

    def _weight_fusion(self, attn_weights, extern_weight, position_weights, mask, batch_size, vid_len):
        """
        融合注意力权重、外部权重和位置权重
        
        参数:
            attn_weights: 注意力权重，形状为 (batch_size, num_heads, compress_len, vid_len)
            extern_weight: 外部权重，形状为 (batch_size, vid_len)
            position_weights: 位置权重，形状为 (batch_size, compress_len, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            batch_size: 批次大小
            vid_len: 视频长度
            
        返回:
            final_weights: 融合后的权重，形状为 (batch_size, num_heads, compress_len, vid_len)
        """
        # 1. 处理外部权重
        if extern_weight is not None:
            # 确保extern_weight的形状正确
            assert extern_weight.shape == (batch_size, vid_len), \
                f"外部权重形状应为 {(batch_size, vid_len)}, 但得到了 {extern_weight.shape}"
            
            # 扩展外部权重以匹配压缩长度
            # (batch_size, vid_len) -> (batch_size, compress_len, vid_len)
            extern_weight_expanded = extern_weight.unsqueeze(1).expand(-1, self.compress_len, -1)
            
            # 应用掩码
            if mask is not None:
                mask_expanded = mask.unsqueeze(1)  # (batch_size, 1, vid_len)
                extern_weight_expanded = extern_weight_expanded.masked_fill(~mask_expanded.bool(), 0)
                
                # 重新归一化外部权重
                row_sums = extern_weight_expanded.sum(dim=-1, keepdim=True)
                extern_weight_expanded = extern_weight_expanded / (row_sums + 1e-9)
            
            # 再次扩展以匹配多头
            # (batch_size, compress_len, vid_len) -> (batch_size, 1, compress_len, vid_len)
            extern_weight_expanded = extern_weight_expanded.unsqueeze(1)
        else:
            # 如果没有外部权重，设为0，不参与融合
            extern_weight_expanded = torch.zeros_like(attn_weights)
        
        # 2. 处理位置权重
        # (batch_size, compress_len, vid_len) -> (batch_size, 1, compress_len, vid_len)
        position_weights_expanded = position_weights.unsqueeze(1)
        
        # 3. 融合三种权重
        final_weights = (
            self.attn_weight_alpha * attn_weights + 
            self.extern_weight_alpha * extern_weight_expanded + 
            self.position_weight_alpha * position_weights_expanded
        )
        
        # 确保权重总和为1
        final_weights_sum = final_weights.sum(dim=-1, keepdim=True)
        final_weights = final_weights / (final_weights_sum + 1e-9)
        
        return final_weights

    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的有效掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 1. 计算位置权重
        position_weights, need_compress, original_indices = self._compute_position_weights(
            mask, batch_size, vid_len)
        
        # 2. 对于需要压缩的样本，使用注意力机制
        if need_compress.any():
            # 获取注意力权重和中间结果
            attn_weights, query, v = self.before_attn(Fv, mask)
            
            # 融合三种权重
            combined_weights = self._weight_fusion(
                attn_weights, extern_weight, position_weights, mask, batch_size, vid_len)
            
            # 使用融合后的权重计算加权和
            weighted_v = torch.matmul(combined_weights, v)
            
            # 后续处理得到压缩后的特征
            Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        # 特殊情况，全部样本的有效长度都小于compress_len，这时就没有压缩的必要了
        else:
            # 所有样本都不需要压缩，只复制前压缩长度个特征
            Fv_compress = torch.zeros(batch_size, self.compress_len, Fv.size(2), device=device)
            compress_mask = torch.zeros(batch_size, self.compress_len, dtype=torch.bool, device=device)
            
            # 针对每个不需要压缩的样本
            for i in range(batch_size):
                valid_len = min(self._get_valid_lengths(mask)[i].item(), self.compress_len)
                
                # 从原始特征中复制有效部分
                indices = original_indices[i, :valid_len]
                Fv_compress[i, :valid_len] = Fv[i, indices]
                
                # 设置有效掩码
                compress_mask[i, :valid_len] = True
        
        return Fv_compress, compress_mask

# 1. 计算attention前，先把ex_weight乘到KV上
class CompressorWithWeightedKV(CompressorWithExternalWeights):
    """
    在计算注意力之前，将外部权重乘到Key和Value上的视频压缩模块。
    
    方案1: 在计算attention前，先把extern_weight乘到KV上，然后再进行self-attention计算
    """
    
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            combined_weights: 注意力权重，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 如果提供了外部权重，直接将其应用到Fv上
        if extern_weight is not None:
            # 扩展形状以便乘法操作: (batch_size, vid_len) -> (batch_size, vid_len, 1)
            weighted_mask = extern_weight.unsqueeze(-1)
            
            # 将权重直接应用到Fv
            weighted_Fv = Fv * weighted_mask
        else:
            # 如果没有外部权重，使用原始的Fv
            weighted_Fv = Fv
        
        # 使用加权后的特征进行注意力计算
        attn_weights, query, v = self.before_attn(weighted_Fv, mask)
        
        # 计算加权和
        weighted_v = torch.matmul(attn_weights, v)
        
        # 后续处理
        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        
        return Fv_compress, compress_mask

# 2. attention之后在输出前乘上ex_weight
class CompressorWithPostWeighting(CompressorWithExternalWeights):
    """
    在注意力计算之后，输出前将外部权重应用于结果的视频压缩模块。
    
    方案2: attention之后在输出前乘上ex_weight
    """
    
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            attn_weights: 注意力权重，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, _ = Fv.size()
        device = Fv.device
        
        # 常规的注意力计算
        attn_weights, query, v = self.before_attn(Fv, mask)
        
        # 如果提供了外部权重，修改注意力权重
        if extern_weight is not None:
            # 扩展外部权重以匹配多头注意力的形状
            # (batch_size, vid_len) -> (batch_size, 1, 1, vid_len)
            ext_weight_expanded = extern_weight.unsqueeze(1).unsqueeze(2)
            
            # 将扩展后的权重应用到注意力权重
            # (batch_size, num_heads, compress_len, vid_len) * (batch_size, 1, 1, vid_len)
            weighted_attn = attn_weights * ext_weight_expanded
            
            # 重新归一化
            sum_weights = weighted_attn.sum(dim=-1, keepdim=True)
            weighted_attn = weighted_attn / (sum_weights + 1e-9)
            
            # 使用加权后的注意力计算输出
            weighted_v = torch.matmul(weighted_attn, v)
        else:
            # 如果没有外部权重，使用原始的注意力权重
            weighted_v = torch.matmul(attn_weights, v)
        
        # 后续处理
        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        
        return Fv_compress, compress_mask

# 3. 平滑机制，在fi直接乘权重的基础上，再加上一个favg*（1-si）
class CompressorWithSmoothMechanism(CompressorWithExternalWeights):
    """
    具有平滑机制的视频压缩模块：fi = fi * si + favg * (1-si)
    
    方案3: 平滑机制，在fi直接乘权重的基础上，再加上一个favg*（1-si）
    即fi = fi * si + favg*（1-si）
    si即每个视频特征上的外部权重，favg是一个视频在有效位的特征平均值
    """
    
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重（si），形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)，True表示有效位置
            weight_alpha: 注意力权重的权重，1-weight_alpha为外部权重的权重
            
        返回:
            Fv_compress: 压缩后的视频特征，形状为 (batch_size, compress_len, dimension)
            combined_weights: 注意力权重，形状为 (batch_size, compress_len, vid_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 计算特征平均值 (favg)
        if mask is not None:
            # 创建掩码来排除无效位置: (batch_size, vid_len, 1)
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, dimension)
            
            # 计算有效位置的平均值
            # 首先，将无效位置置为0
            masked_Fv = Fv * mask_expanded.float()
            
            # 然后，计算每个batch样本中有效特征的平均值
            # 对于每个batch样本，计算有效位置的数量
            valid_counts = mask.float().sum(dim=1, keepdim=True)  # (batch_size, 1)
            
            # 计算有效位置的特征总和
            sum_features = masked_Fv.sum(dim=1)  # (batch_size, dimension)
            
            # 计算平均值，避免除以0
            favg = sum_features / (valid_counts + 1e-9)  # (batch_size, dimension)
        else:
            # 如果没有掩码，计算所有位置的平均值
            favg = Fv.mean(dim=1)  # (batch_size, dimension)
        
        # 如果提供了外部权重 (si)
        if extern_weight is not None:
            # 确保extern_weight形状正确
            assert extern_weight.shape == (batch_size, vid_len), \
                f"外部权重形状应为 {(batch_size, vid_len)}, 但得到了 {extern_weight.shape}"
            
            # 应用掩码（如果提供）
            if mask is not None:
                si = extern_weight.clone()
                si = si.masked_fill(~mask.bool(), 0)
            else:
                si = extern_weight
            
            # 实现平滑机制: fi = fi * si + favg * (1-si)
            # 扩展si和favg以便于乘法操作
            si_expanded = si.unsqueeze(-1)  # (batch_size, vid_len, 1)
            favg_expanded = favg.unsqueeze(1)  # (batch_size, 1, dimension)
            
            # 计算平滑后的特征
            smoothed_Fv = Fv * si_expanded + favg_expanded * (1 - si_expanded)
        else:
            # 如果没有外部权重，使用原始的Fv
            smoothed_Fv = Fv
        
        # 使用平滑后的特征进行注意力计算
        attn_weights, query, v = self.before_attn(smoothed_Fv, mask)
        
        # 计算加权和
        weighted_v = torch.matmul(attn_weights, v)
        
        # 后续处理
        Fv_compress, compress_mask = self.after_attn(weighted_v, query, batch_size, device)
        
        return Fv_compress, compress_mask

# 4. 残差，将压缩结果Fs和最近compress_len个long memory相加
class ResidualCompressor(CompressorWithExternalWeights):
    def __init__(self, compress_len, dimension, num_heads=8, dropout=0.1, weight_alpha=0.5, residual_weight=0.5):
        super().__init__(compress_len=compress_len, dimension=dimension, weight_alpha=weight_alpha)
        self.residual_weight = residual_weight
        
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播：对Fv进行压缩后与最近的有效帧做残差连接。
        
        Args:
            Fv: 视频特征 [batch_size, vid_len, dimension]
            extern_weight: 外部权重（可选）
            mask: 掩码 [batch_size, vid_len]
        Returns:
            Fv_residual: 残差结果 [batch_size, compress_len, dimension]
            compress_mask: 压缩掩码 [batch_size, compress_len]
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 压缩
        Fv_compress, compress_mask = super().forward(Fv, extern_weight, mask)

        # 提取每个 batch 的有效特征（取最近的compress_len个）
        bool_mask = mask.bool()
        Fo = []
        for batch in range(batch_size):
            Fo.append(Fv[batch][bool_mask[batch]])
            Fo[batch] = Fo[batch][-self.compress_len:]

        # 将Fo中长度第一维长度小于compress_len的填充到compress_len
        padded_Fo = torch.zeros(batch_size, self.compress_len, dimension, device=device)
        for i in range(batch_size):
            if Fo[i].shape[0] < self.compress_len:
                padded_Fo[i] = F.pad(Fo[i], (0, 0, 0, self.compress_len - Fo[i].shape[0]), "constant", 0)
            else:
                padded_Fo[i] = Fo[i]

        # 残差连接
        Fv_residual = padded_Fo + self.residual_weight * Fv_compress

        # 对于每个batch的结果，如果Fo长度小于compress_len，则直接返回未压缩的Fv，
        # 并将mask根据Fo的长度重新设置：前len(Fo[i])个为有效，其余为无效
        for i in range(batch_size):
            if Fo[i].shape[0] < self.compress_len:
                Fv_residual[i] = Fv[i][:self.compress_len]
                compress_mask[i] = torch.zeros(self.compress_len, dtype=torch.bool, device=device)
                compress_mask[i][:len(Fo[i])] = True

        return Fv_residual, compress_mask

# 5. 残差+attn
class CrossAttentionResidualCompressor(CompressorWithExternalWeights):
    """
    使用交叉注意力机制的残差压缩模块
    
    在压缩结果Fs的基础上，取最近的compress_len个记忆帧Fo，
    通过 Fo + δ * attn(Fo, Fs) 的方式进行融合
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        mask: 有效位置的掩码，形状为 (batch_size, vid_len)
        
    输出:
        Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5, attn_weight=0.5):
        """
        初始化交叉注意力残差压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            weight_alpha: 注意力权重和外部权重的融合比例
            attn_weight: 注意力结果的权重系数δ
        """
        super().__init__(dimension, compress_len, num_heads, dropout, weight_alpha)
        self.attn_weight = attn_weight  # δ参数，控制注意力结果的权重
        
        # 交叉注意力层的投影矩阵
        self.cross_q_proj = nn.Linear(dimension, dimension)
        self.cross_k_proj = nn.Linear(dimension, dimension)
        self.cross_v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.cross_out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.cross_dropout = nn.Dropout(dropout)
        
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 1. 首先获取压缩的视频特征 Fs (Fv_compress)
        Fs, compress_mask = super().forward(Fv, extern_weight, mask)
        
        # 2. 提取每个batch的最近compress_len个有效特征作为Fo
        bool_mask = mask.bool() if mask is not None else torch.ones(batch_size, vid_len, dtype=torch.bool, device=device)
        Fo_list = []
        for batch in range(batch_size):
            # 获取当前批次的有效特征
            valid_features = Fv[batch][bool_mask[batch]]
            # 取最后compress_len个
            Fo_list.append(valid_features[-self.compress_len:])
        
        # 3. 创建Fo的填充张量和Fo的掩码
        padded_Fo = torch.zeros(batch_size, self.compress_len, dimension, device=device)
        fo_mask = torch.zeros(batch_size, self.compress_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            fo_len = Fo_list[i].shape[0]
            padded_Fo[i, :fo_len] = Fo_list[i]
            fo_mask[i, :fo_len] = True
        
        # 4. 对于batch中Fo长度小于compress_len的样本，直接使用原始特征
        # 创建结果变量，初始为Fs
        # Fv_residual = Fs.clone()
        # 创建结果变量，初始为形状和Fs相同的0
        Fv_residual = torch.zeros_like(Fs)
        special_cases = []

        for i in range(batch_size):
            if Fo_list[i].shape[0] < self.compress_len:
                # 如果Fo长度小于compress_len，直接使用原始特征
                # Fv_residual[i, :Fo_list[i].shape[0]] = Fv[i][bool_mask[i]][-Fo_list[i].shape[0]:]
                # compress_mask[i] = torch.zeros(self.compress_len, dtype=torch.bool, device=device)
                # compress_mask[i][:Fo_list[i].shape[0]] = True

                Fv_residual[i] = padded_Fo[i]
                compress_mask[i] = fo_mask[i]

                special_cases.append(i)
        
        # 如果所有batch都是特殊情况，直接返回结果
        if len(special_cases) == batch_size:
            return Fv_residual, compress_mask
        
        # 创建一个掩码，标记哪些batch需要进行交叉注意力计算
        batch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for idx in special_cases:
            batch_mask[idx] = False
        
        # 5. 对其余样本执行交叉注意力: Fo作为query，Fs作为key和value
        if batch_mask.any():
            # 筛选需要进行注意力计算的batch
            filtered_Fo = padded_Fo[batch_mask]
            filtered_Fs = Fs[batch_mask]
            filtered_fo_mask = fo_mask[batch_mask]
            
            # 交叉注意力计算
            # 5.1 投影query, key, value
            q = self.cross_q_proj(filtered_Fo)  # (filtered_batch_size, compress_len, dimension)
            k = self.cross_k_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            v = self.cross_v_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.2 分割多头
            filtered_batch_size = filtered_Fo.size(0)
            head_dim = dimension // self.num_heads
            
            # 重塑为多头格式
            q = q.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            
            # 5.3 计算注意力分数
            # (filtered_batch_size, num_heads, compress_len, head_dim) @ (filtered_batch_size, num_heads, head_dim, compress_len)
            # -> (filtered_batch_size, num_heads, compress_len, compress_len)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            
            # 应用Fo的掩码（确保填充部分不参与注意力计算）
            if filtered_fo_mask is not None:
                # 扩展掩码以适应多头注意力的形状
                # (filtered_batch_size, compress_len) -> (filtered_batch_size, 1, compress_len, 1)
                q_mask = filtered_fo_mask.unsqueeze(1).unsqueeze(-1)
                # 在无效查询位置应用大的负值
                scores = scores.masked_fill(~q_mask.bool(), -1e9)
            
            # 应用softmax得到注意力权重
            attn_weights = F.softmax(scores, dim=-1)  # (filtered_batch_size, num_heads, compress_len, compress_len)
            attn_weights = self.cross_dropout(attn_weights)
            
            # 5.4 计算加权和
            # (filtered_batch_size, num_heads, compress_len, compress_len) @ (filtered_batch_size, num_heads, compress_len, head_dim)
            # -> (filtered_batch_size, num_heads, compress_len, head_dim)
            weighted_v = torch.matmul(attn_weights, v)
            
            # 5.5 合并多头结果
            weighted_v = weighted_v.transpose(1, 2).contiguous()  # (filtered_batch_size, compress_len, num_heads, head_dim)
            weighted_v = weighted_v.view(filtered_batch_size, self.compress_len, dimension)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.6 应用输出投影
            attn_output = self.cross_out_proj(weighted_v)  # (filtered_batch_size, compress_len, dimension)
            
            # 6. 计算残差连接: Fo + δ * attn(Fo, Fs)
            # 对于Fo的无效(padding)部分，我们不应用残差
            filtered_residual = filtered_Fo + self.attn_weight * attn_output

            # 7. 将结果放回到原始batch中
            Fv_residual[batch_mask] = filtered_residual

        return Fv_residual, compress_mask

class CrossAttentionResidualCompressorWithQuery(TextGuidedCompressor):
    """
    使用交叉注意力机制的残差压缩模块
    
    在压缩结果Fs的基础上，取最近的compress_len个记忆帧Fo，
    通过 Fo + δ * attn(Fo, Fs) 的方式进行融合
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        mask: 有效位置的掩码，形状为 (batch_size, vid_len)
        
    输出:
        Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(
        self,
        dimension,
        compress_len,
        num_heads=8,
        dropout=0.1,
        weight_alpha=0.5,
        attn_weight=0.5,
        compressor_output_mode="legacy_fuse",
        ablate_delta_stream: bool = False,
        use_saliency_bias: bool = True,
        use_distance_bias: bool = True,
    ):
        """
        初始化交叉注意力残差压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 压缩后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            weight_alpha: 注意力权重和外部权重的融合比例
            attn_weight: 注意力结果的权重系数δ
        """
        super().__init__(
            t_dim=dimension,
            v_dim=dimension,
            hidden_dim=dimension,
            compress_len=compress_len,
            num_heads=num_heads,
            dropout=dropout,
            weight_alpha=weight_alpha,
            compressor_output_mode=compressor_output_mode,
            ablate_delta_stream=ablate_delta_stream,
            use_saliency_bias=use_saliency_bias,
            use_distance_bias=use_distance_bias,
        )
        self.attn_weight = attn_weight  # δ参数，控制注意力结果的权重
        self.num_heads = num_heads
        
        # 交叉注意力层的投影矩阵
        self.cross_q_proj = nn.Linear(dimension, dimension)
        self.cross_k_proj = nn.Linear(dimension, dimension)
        self.cross_v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.cross_out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.cross_dropout = nn.Dropout(dropout)
        
    @countTime
    def forward(
        self,
        Ft,
        text_mask,
        Fv,
        extern_weight=None,
        vid_mask=None,
        prev_saliency=None,
        frame_index=None,
        center_index=None,
    ):
        """
        前向传播
        
        参数:
            Ft: 文本特征，形状为 (batch_size, num_token, t_dim)
            text_mask: 文本掩码，形状为 (batch_size, num_token)
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            prev_saliency: 历史显著性先验，形状为 (batch_size, vid_len)
            frame_index: long memory 的全局帧索引，形状为 (batch_size, vid_len)
            center_index: 当前片段中心索引，形状为 (batch_size,)
            Fo_list: 每个batch的最近compress_len个有效特征,一个长为bsz的列表
            
        返回:
            Fv_residual: 融合后的压缩视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        
        # 1. 首先获取压缩的视频特征 Fs (Fv_compress)
        Fs, _ = super().forward(
            Ft,
            Fv,
            vid_mask=vid_mask,
            text_mask=text_mask,
            extern_weight=extern_weight,
            prev_saliency=prev_saliency,
            frame_index=frame_index,
            center_index=center_index,
        )
        is_dual_intra_output = isinstance(Fs, dict)
        if is_dual_intra_output:
            Fs_fused = Fs["fused"]
        else:
            Fs_fused = Fs
        compress_mask = torch.ones(
            batch_size,
            self.compress_len,
            dtype=torch.bool,
            device=device,
        )

        bool_mask = vid_mask.bool() if vid_mask is not None else torch.ones(batch_size, vid_len, dtype=torch.bool, device=device)

        # 2. 提取每个batch的最近compress_len个有效特征作为Fo
        Fo_list = []
        for batch in range(batch_size):
            # 获取当前批次的有效特征
            valid_features = Fv[batch][bool_mask[batch]]
            # 取最后compress_len个
            Fo_list.append(valid_features[-self.compress_len:])
        
        # 3. 创建Fo的填充张量和Fo的掩码
        padded_Fo = torch.zeros(batch_size, self.compress_len, dimension, device=device)
        fo_mask = torch.zeros(batch_size, self.compress_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            fo_len = Fo_list[i].shape[0]
            # 如果Fo长度小于compress_len，进行填充
            if fo_len < self.compress_len:
                padded_Fo[i, :fo_len] = Fo_list[i]
                fo_mask[i, :fo_len] = True
            else:
                padded_Fo[i] = Fo_list[i]
                fo_mask[i] = torch.ones(self.compress_len, dtype=torch.bool, device=device)
        
        # 4. 对于batch中Fo长度小于等于compress_len的样本，直接使用原始特征
        # 创建结果变量，初始为Fs
        Fv_residual = Fs_fused.clone()
        special_cases = []
        
        for i in range(batch_size):
            if Fo_list[i].shape[0] < self.compress_len:
                # 如果Fo长度小于等于compress_len，直接使用原始特征
                Fv_residual[i, :Fo_list[i].shape[0]] = Fv[i][bool_mask[i]][-Fo_list[i].shape[0]:]
                compress_mask[i] = torch.zeros(self.compress_len, dtype=torch.bool, device=device)
                compress_mask[i][:Fo_list[i].shape[0]] = True
                special_cases.append(i)
        
        # 如果所有batch都是特殊情况，直接返回结果
        if len(special_cases) == batch_size:
            return Fv_residual, compress_mask
        
        # 创建一个掩码，标记哪些batch需要进行交叉注意力计算
        batch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for idx in special_cases:
            batch_mask[idx] = False
        
        # 5. 对其余样本执行交叉注意力: Fo作为query，Fs作为key和value
        if batch_mask.any():
            # 筛选需要进行注意力计算的batch
            filtered_Fo = padded_Fo[batch_mask]
            filtered_Fs = Fs_fused[batch_mask]
            filtered_fo_mask = fo_mask[batch_mask]
            
            # 交叉注意力计算
            # 5.1 投影query, key, value
            q = self.cross_q_proj(filtered_Fo)  # (filtered_batch_size, compress_len, dimension)
            k = self.cross_k_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            v = self.cross_v_proj(filtered_Fs)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.2 分割多头
            filtered_batch_size = filtered_Fo.size(0)
            head_dim = dimension // self.num_heads
            
            # 重塑为多头格式
            q = q.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            
            # 5.3 计算注意力分数
            # (filtered_batch_size, num_heads, compress_len, head_dim) @ (filtered_batch_size, num_heads, head_dim, compress_len)
            # -> (filtered_batch_size, num_heads, compress_len, compress_len)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            
            # 应用Fo的掩码（确保填充部分不参与注意力计算）
            if filtered_fo_mask is not None:
                # 扩展掩码以适应多头注意力的形状
                # (filtered_batch_size, compress_len) -> (filtered_batch_size, 1, compress_len, 1)
                q_mask = filtered_fo_mask.unsqueeze(1).unsqueeze(-1)
                # 在无效查询位置应用大的负值
                scores = scores.masked_fill(~q_mask.bool(), -1e9)
            
            # 应用softmax得到注意力权重
            attn_weights = F.softmax(scores, dim=-1)  # (filtered_batch_size, num_heads, compress_len, compress_len)
            attn_weights = self.cross_dropout(attn_weights)
            
            # 5.4 计算加权和
            # (filtered_batch_size, num_heads, compress_len, compress_len) @ (filtered_batch_size, num_heads, compress_len, head_dim)
            # -> (filtered_batch_size, num_heads, compress_len, head_dim)
            weighted_v = torch.matmul(attn_weights, v)
            
            # 5.5 合并多头结果
            weighted_v = weighted_v.transpose(1, 2).contiguous()  # (filtered_batch_size, compress_len, num_heads, head_dim)
            weighted_v = weighted_v.view(filtered_batch_size, self.compress_len, dimension)  # (filtered_batch_size, compress_len, dimension)
            
            # 5.6 应用输出投影
            attn_output = self.cross_out_proj(weighted_v)  # (filtered_batch_size, compress_len, dimension)
            
            # 6. 计算残差连接: Fo + δ * attn(Fo, Fs)
            filtered_residual = (1 - self.attn_weight) * filtered_Fo + self.attn_weight * attn_output
            
            # 7. 将结果放回到原始batch中
            Fv_residual[batch_mask] = filtered_residual

        if is_dual_intra_output:
            dual_intra_output = {
                "static": Fs["static"],
                "delta": Fs["delta"],
                "fused": Fv_residual,
            }
            return dual_intra_output, compress_mask
        return Fv_residual, compress_mask

class OnlineCrossAttentionResidualCompressor(CrossAttentionResidualCompressorWithQuery):
    """
    在线优化版本的CrossAttentionResidualCompressorWithQuery
    
    优化点：
    1. 缓存双分支压缩器的K/V投影（处理历史帧Fv）
    2. 维护Fo的滑动窗口（最近compress_len个帧）
    3. 交叉注意力部分保持不变（因为Query和Key都在变化）
    """
    
    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, 
                 weight_alpha=0.5, attn_weight=0.5):
        super().__init__(dimension, compress_len, num_heads, dropout, 
                        weight_alpha, attn_weight)
        
        # === 在线模式相关 ===
        self.online_mode = False
        self.max_history_len = None  # 历史帧最大长度M_h

        self.dimension = dimension
        
        # 1. 双分支压缩器的缓存
        # text_enhanced分支的缓存
        self.cached_text_enhanced_k = None
        self.cached_text_enhanced_v = None
        
        # direct分支的缓存
        self.cached_direct_k = None
        self.cached_direct_v = None

        # 缓存文本增强后的历史帧（当前由update_text_enhanced_cache_incremental使用）
        self.cached_Fv_enhanced = None  # (batch, max_history_len, dim)

        # 缓存文本增强过程中的注意力分数
        self.cached_text_enhanced_attn_scores = None  # (batch, max_history_len)
        
        # 2. Fo滑动窗口缓存（最近compress_len个有效帧）
        self.cached_Fo = None  # (batch_size, compress_len, dimension)
        self.cached_Fo_mask = None  # (batch_size, compress_len)
        self.Fo_valid_len = None  # 每个batch当前Fo的有效长度，(batch_size,)
        
        # 当前缓存的有效长度
        self.cache_valid_len = 0
        
    def enable_online_mode(self, max_history_len, batch_size, device):
        """
        启用在线推理优化模式
        
        Args:
            max_history_len: 历史帧最大长度M_h
            batch_size: 批次大小
            device: 设备
        """
        self.online_mode = True
        self.max_history_len = max_history_len
        self.reset_cache(batch_size, device)
        
    def disable_online_mode(self):
        """禁用在线模式，清空所有缓存"""
        self.online_mode = False
        self.cached_text_enhanced_k = None
        self.cached_text_enhanced_v = None
        self.cached_direct_k = None
        self.cached_direct_v = None
        self.cached_Fo = None
        self.cached_Fo_mask = None
        self.Fo_valid_len = None
        self.cache_valid_len = 0
        
    def reset_cache(self, batch_size, device):
        """重置所有缓存"""
        dimension = self.dimension
        
        # 双分支压缩器的K/V缓存
        self.cached_text_enhanced_k = torch.zeros(
            batch_size, self.max_history_len, dimension, device=device
        )
        self.cached_text_enhanced_v = torch.zeros(
            batch_size, self.max_history_len, dimension, device=device
        )
        self.cached_direct_k = torch.zeros(
            batch_size, self.max_history_len, dimension, device=device
        )
        self.cached_direct_v = torch.zeros(
            batch_size, self.max_history_len, dimension, device=device
        )
        
        # Fo滑动窗口缓存
        self.cached_Fo = torch.zeros(
            batch_size, self.compress_len, dimension, device=device
        )
        self.cached_Fo_mask = torch.zeros(
            batch_size, self.compress_len, dtype=torch.bool, device=device
        )
        self.Fo_valid_len = torch.zeros(batch_size, dtype=torch.long, device=device)

        # # 文本增强过程中的注意力分数
        # self.cached_text_enhanced_attn_scores = torch.zeros(
        #     batch_size, self.max_history_len, device=device
        # )
        
        self.cache_valid_len = 0
    
    def update_Fo_sliding_window(self, new_frame_feat, batch_idx=None):
        """
        更新Fo的滑动窗口
        
        Args:
            new_frame_feat: 新帧特征 (batch_size, 1, dimension) 或 (1, dimension)
            batch_idx: 如果指定，只更新特定batch；否则更新所有batch
        """
        if new_frame_feat.dim() == 2:
            new_frame_feat = new_frame_feat.unsqueeze(1)
        
        batch_size = new_frame_feat.size(0)
        
        if batch_idx is not None:
            # 只更新指定batch
            batch_indices = [batch_idx]
        else:
            # 更新所有batch
            batch_indices = range(batch_size)
        
        for idx in batch_indices:
            current_len = self.Fo_valid_len[idx].item()
            
            if current_len < self.compress_len:
                # Fo未满，直接追加
                self.cached_Fo[idx, current_len] = new_frame_feat[idx, 0]
                self.cached_Fo_mask[idx, current_len] = True
                self.Fo_valid_len[idx] += 1
            else:
                # Fo已满，FIFO：移除最旧的，添加最新的
                self.cached_Fo[idx, :-1] = self.cached_Fo[idx, 1:].clone()
                self.cached_Fo[idx, -1] = new_frame_feat[idx, 0]
                # mask保持全True（因为已经满了）
    
    def update_text_enhanced_cache_incremental(self, new_frame_feat, Ft, text_mask):
        """
        只对新帧计算文本增强，更新 Fv_enhanced 缓存（FIFO），但不更新 cache_valid_len
        
        Returns:
            new_Fv_enhanced: 新计算出的增强帧特征 (batch, 1, dim)
        """
        # 1. 只增强新帧
        new_Fv_enhanced = self.text_guided_video_attention(
            Ft, new_frame_feat, vid_mask=None, text_mask=text_mask
        )  # (batch, 1, dim)
        
        # 2. 更新缓存（FIFO）
        #  注意：这里使用 self.cache_valid_len 作为当前的插入位置
        if self.cache_valid_len < self.max_history_len:
            self.cached_Fv_enhanced[:, self.cache_valid_len] = new_Fv_enhanced.squeeze(1)
            #  移除 self.cache_valid_len += 1，留给 update_compressor_cache_incremental 统一管理
        else:
            self.cached_Fv_enhanced = torch.cat([
                self.cached_Fv_enhanced[:, 1:],
                new_Fv_enhanced
            ], dim=1)
        
        return new_Fv_enhanced

    def update_compressor_cache_incremental(self, new_frame_feat, Ft, text_mask):
        """
        增量更新双分支压缩器的K/V缓存
        
        Args:
            new_frame_feat: 新帧特征 (batch_size, 1, dimension)
            Ft: 文本特征 (batch_size, num_token, t_dim)
            text_mask: 文本掩码 (batch_size, num_token)
        """
        # 1. 计算text_enhanced分支的新帧K/V
        # 首先通过text_guided_video_attention增强新帧
        Fv_enhanced_new = self.text_guided_video_attention(
            Ft, new_frame_feat, vid_mask=None, text_mask=text_mask
        )  # (batch_size, 1, dimension)

        # Fv_enhanced_new = self.update_text_enhanced_cache_incremental(new_frame_feat, Ft, text_mask)
        
        # 然后投影为K/V（使用text_enhanced_memory_compressor的投影层）
        new_k_text = self.text_enhanced_memory_compressor.k_proj(Fv_enhanced_new)
        new_v_text = self.text_enhanced_memory_compressor.v_proj(Fv_enhanced_new)
        
        # 2. 计算direct分支的新帧K/V
        new_k_direct = self.direct_memory_compressor.k_proj(new_frame_feat)
        new_v_direct = self.direct_memory_compressor.v_proj(new_frame_feat)
        
        # 3. 更新缓存（FIFO）
        if self.cache_valid_len < self.max_history_len:
            # 缓存未满，直接追加
            self.cached_text_enhanced_k[:, self.cache_valid_len] = new_k_text.squeeze(1)
            self.cached_text_enhanced_v[:, self.cache_valid_len] = new_v_text.squeeze(1)
            self.cached_direct_k[:, self.cache_valid_len] = new_k_direct.squeeze(1)
            self.cached_direct_v[:, self.cache_valid_len] = new_v_direct.squeeze(1)
            self.cache_valid_len += 1
        else:
            # 缓存已满，FIFO
            self.cached_text_enhanced_k = torch.cat([
                self.cached_text_enhanced_k[:, 1:],
                new_k_text
            ], dim=1)
            self.cached_text_enhanced_v = torch.cat([
                self.cached_text_enhanced_v[:, 1:],
                new_v_text
            ], dim=1)
            self.cached_direct_k = torch.cat([
                self.cached_direct_k[:, 1:],
                new_k_direct
            ], dim=1)
            self.cached_direct_v = torch.cat([
                self.cached_direct_v[:, 1:],
                new_v_direct
            ], dim=1)
    
    def forward_online(self, new_frame_feat, Ft, text_mask, vid_mask=None, extern_weight=None):
        """
        在线推理模式的前向传播
        
        Args:
            new_frame_feat: 新到达的帧特征 (batch_size, 1, dimension)
            Ft: 文本特征 (batch_size, num_token, t_dim)
            text_mask: 文本掩码 (batch_size, num_token)
            vid_mask: 视频掩码（在线模式下通常不需要）
            extern_weight: 外部权重（在线模式下通常不需要）
        
        Returns:
            Fv_residual: 融合后的压缩特征 (batch_size, compress_len, dimension)
            compress_mask: 压缩后的掩码 (batch_size, compress_len)
        """
        batch_size = new_frame_feat.size(0)
        device = new_frame_feat.device

        # 检查并调整缓存大小
        if self.cached_Fo is None or self.cached_Fo.size(0) != batch_size:
            self.reset_cache(batch_size, device)
        
        # === Step 1: 更新所有缓存 ===
        # 1.1 更新Fo滑动窗口
        self.update_Fo_sliding_window(new_frame_feat)
        
        # 1.2 更新双分支压缩器的K/V缓存
        self.update_compressor_cache_incremental(new_frame_feat, Ft, text_mask)
        
        # === Step 2: 使用缓存计算双分支压缩结果Fs ===
        # 2.1 获取当前有效的K/V
        valid_k_text = self.cached_text_enhanced_k[:, :self.cache_valid_len]
        valid_v_text = self.cached_text_enhanced_v[:, :self.cache_valid_len]
        valid_k_direct = self.cached_direct_k[:, :self.cache_valid_len]
        valid_v_direct = self.cached_direct_v[:, :self.cache_valid_len]
        
        # 2.2 计算text_enhanced分支的压缩结果
        Fv_compress_enhanced = self._compute_compressed_from_cached_kv(
            valid_k_text, valid_v_text,
            self.text_enhanced_memory_compressor
        )
        
        # 2.3 计算direct分支的压缩结果
        Fv_compress_direct = self._compute_compressed_from_cached_kv(
            valid_k_direct, valid_v_direct,
            self.direct_memory_compressor
        )
        
        # 2.4 门控融合双分支
        direct_gate_value = self.direct_gate(Fv_compress_direct.mean(1, keepdim=True))
        text_enhanced_gate_value = self.text_enhanced_gate(
            Fv_compress_enhanced.mean(1, keepdim=True)
        )
        
        Fs = self.norm(
            direct_gate_value * Fv_compress_direct + 
            text_enhanced_gate_value * Fv_compress_enhanced
        )
        
        # === Step 3: 交叉注意力：Fo与Fs ===
        # 注意：这部分无法缓存，因为Fo和Fs都在变化
        
        # 3.1 获取当前Fo
        Fo = self.cached_Fo.clone()
        fo_mask = self.cached_Fo_mask.clone()
        
        # 3.2 检查是否有batch的Fo长度小于compress_len（特殊情况）
        special_cases = (self.Fo_valid_len < self.compress_len).nonzero(as_tuple=True)[0]
        
        if len(special_cases) == batch_size:
            # 所有batch都是特殊情况，直接返回Fo
            compress_mask = fo_mask
            return Fo, compress_mask
        
        # 3.3 对于Fo长度>=compress_len的batch，执行交叉注意力
        batch_mask = self.Fo_valid_len >= self.compress_len
        
        if batch_mask.any():
            filtered_Fo = Fo[batch_mask]
            filtered_Fs = Fs[batch_mask]
            filtered_fo_mask = fo_mask[batch_mask]
            
            # 交叉注意力计算（与原实现相同）
            filtered_residual = self._cross_attention_residual(
                filtered_Fo, filtered_Fs, filtered_fo_mask
            )
            
            # 将结果放回
            Fo[batch_mask] = filtered_residual
        
        # 3.4 对于特殊情况的batch，使用原始Fo
        # （已经在Fo中，无需额外操作）
        
        compress_mask = fo_mask
        return Fo, compress_mask
    
    def _compute_compressed_from_cached_kv(self, cached_k, cached_v, compressor):
        """
        从缓存的K/V计算压缩结果（核心优化：避免重复投影）
        
        Args:
            cached_k: 缓存的K (batch_size, valid_len, dimension)
            cached_v: 缓存的V (batch_size, valid_len, dimension)
            compressor: 压缩器实例（SimpleCompressor或CompressorWithExternalWeights）
        
        Returns:
            compressed_feat: 压缩后的特征 (batch_size, compress_len, dimension)
        """
        batch_size = cached_k.size(0)
        valid_len = cached_k.size(1)
        
        # 1. Query投影（Query是可学习参数，不需要缓存）
        query = compressor.query.expand(batch_size, -1, -1)
        q = compressor.q_proj(query)  # (batch_size, compress_len, dimension)
        
        # 2. 直接使用缓存的K和V（跳过k_proj和v_proj）
        k = cached_k
        v = cached_v
        
        # 3. 多头注意力计算（与原实现相同）
        head_dim = self.dimension // self.num_heads
        
        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = compressor.attn_dropout(attn_weights)
        
        weighted_v = torch.matmul(attn_weights, v)
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        weighted_v = weighted_v.view(batch_size, self.compress_len, self.dimension)
        
        # 4. 输出投影
        compressed_feat = compressor.out_proj(weighted_v)
        
        # 5. 残差连接和前馈网络（如果有的话）
        compressed_feat = compressor.norm(query + compressed_feat)
        ff_output = compressor.feed_forward(compressed_feat)
        compressed_feat = compressor.final_norm(compressed_feat + ff_output)
        
        return compressed_feat
    
    def _cross_attention_residual(self, Fo, Fs, fo_mask):
        """
        交叉注意力残差连接（与原实现相同，提取出来便于复用）
        
        Args:
            Fo: 最近帧 (filtered_batch_size, compress_len, dimension)
            Fs: 压缩结果 (filtered_batch_size, compress_len, dimension)
            fo_mask: Fo的掩码 (filtered_batch_size, compress_len)
        
        Returns:
            residual: 残差结果 (filtered_batch_size, compress_len, dimension)
        """
        filtered_batch_size = Fo.size(0)
        head_dim = self.dimension // self.num_heads
        
        # 投影Q, K, V
        q = self.cross_q_proj(Fo)
        k = self.cross_k_proj(Fs)
        v = self.cross_v_proj(Fs)
        
        # 多头分割
        q = q.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        if fo_mask is not None:
            q_mask = fo_mask.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(~q_mask.bool(), -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.cross_dropout(attn_weights)
        
        weighted_v = torch.matmul(attn_weights, v)
        weighted_v = weighted_v.transpose(1, 2).contiguous()
        weighted_v = weighted_v.view(filtered_batch_size, self.compress_len, self.dimension)
        
        attn_output = self.cross_out_proj(weighted_v)
        
        # 残差连接
        residual = (1 - self.attn_weight) * Fo + self.attn_weight * attn_output
        
        return residual
    
    def forward(self, Ft, text_mask, Fv, extern_weight=None, vid_mask=None):
        """
        前向传播：自动判断使用标准模式还是在线模式
        
        Args:
            Ft: 文本特征
            text_mask: 文本掩码
            Fv: 视频特征（标准模式）或新帧特征（在线模式）
            extern_weight: 外部权重
            vid_mask: 视频掩码
        """
        if self.online_mode:
            # 在线模式：Fv被视为新到达的帧
            return self.forward_online(Fv[:, -1, :].unsqueeze(1), Ft, text_mask, vid_mask, extern_weight)
        else:
            # 标准模式：调用父类的完整实现
            return super().forward(Ft, text_mask, Fv, extern_weight, vid_mask)

# 7. 拼接，compress_len长度的压缩结果，前半部分为压缩到compress_len/2的压缩结果，后半部分为最近的compress_len/2个视觉特征
class ConcatCompressor(CompressorWithExternalWeights):
    """
    使用拼接的压缩模块
    
    压缩结果共compress_len长度，前半部分为压缩到compress_len/2的压缩结果，
    后半部分为最近的compress_len/2个视觉特征
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        mask: 有效位置的掩码，形状为 (batch_size, vid_len)
        
    输出:
        Fv_concat: 拼接后的视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 拼接后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5):
        """
        初始化拼接压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 拼接后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            weight_alpha: 注意力权重和外部权重的融合比例
        """
        # 确保compress_len是偶数
        if compress_len % 2 != 0:
            raise ValueError("compress_len必须是偶数")
            
        # 调用父类初始化，但压缩长度为compress_len的一半
        super().__init__(dimension, compress_len // 2, num_heads, dropout, weight_alpha)
        
    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            Fv_concat: 拼接后的视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 拼接后的掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        half_compress_len = self.compress_len  # 实际压缩长度是总长度的一半
        full_compress_len = half_compress_len * 2  # 最终长度
        
        # 1. 获取压缩的视频特征 Fs (前半部分)
        Fs, compress_mask_half = super().forward(Fv, extern_weight, mask)
        
        # 2. 提取每个batch的最近compress_len/2个有效特征作为Fo
        bool_mask = mask.bool() if mask is not None else torch.ones(batch_size, vid_len, dtype=torch.bool, device=device)
        Fo_list = []
        actual_fo_lengths = []
        
        for batch in range(batch_size):
            # 获取当前批次的有效特征
            valid_features = Fv[batch][bool_mask[batch]]
            # 记录实际有效特征长度
            valid_len = valid_features.shape[0]
            actual_fo_lengths.append(min(valid_len, half_compress_len))
            # 取最后compress_len/2个
            Fo_list.append(valid_features[-half_compress_len:] if valid_len >= half_compress_len 
                          else valid_features)
        
        # 3. 创建结果变量，初始化为零张量
        Fv_concat = torch.zeros(batch_size, full_compress_len, dimension, device=device)
        full_compress_mask = torch.zeros(batch_size, full_compress_len, dtype=torch.bool, device=device)
        
        # 4. 处理每个batch样本
        for i in range(batch_size):
            # 获取当前批次的有效特征总数
            valid_features_count = bool_mask[i].sum().item()
            
            if valid_features_count < full_compress_len:
                # 特殊情况：有效特征总数小于full_compress_len
                # 直接使用原始的所有有效特征，pad到full_compress_len
                valid_features = Fv[i][bool_mask[i]]
                # 放入结果张量的前valid_features_count位置
                Fv_concat[i, :valid_features_count] = valid_features
                # 设置对应的mask
                full_compress_mask[i, :valid_features_count] = True
                # 其余位置保持为0和False（已初始化为这些值）
            else:
                # 正常情况：前半部分为压缩特征，后半部分为最近的特征
                # 前半部分：压缩特征
                Fv_concat[i, :half_compress_len] = Fs[i]
                full_compress_mask[i, :half_compress_len] = compress_mask_half[i]
                
                # 后半部分：最近的half_compress_len个特征
                Fv_concat[i, half_compress_len:] = Fo_list[i]
                full_compress_mask[i, half_compress_len:] = True
        
        return Fv_concat, full_compress_mask

class CrossAttentionConcatCompressor(CompressorWithExternalWeights):
    """
    使用交叉注意力的拼接压缩模块（改进版）
    
    压缩结果共compress_len长度，前半部分完全使用交叉注意力机制处理的结果 attn(Fo, Fs)，
    交叉注意力用Fo指导对Fs的处理
    后半部分仍为最近的compress_len/2个视觉特征
    
    输入:
        Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
        extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
        mask: 有效位置的掩码，形状为 (batch_size, vid_len)
        
    输出:
        Fv_concat: 拼接后的视频特征，形状为 (batch_size, compress_len, dimension)
        compress_mask: 拼接后的掩码，形状为 (batch_size, compress_len)
    """
    def __init__(self, dimension, compress_len, num_heads=8, dropout=0.1, weight_alpha=0.5):
        """
        初始化拼接压缩模块
        
        参数:
            dimension: 特征维度
            compress_len: 拼接后的视频长度
            num_heads: 多头注意力的头数
            dropout: Dropout率
            weight_alpha: 注意力权重和外部权重的融合比例
        """
        # 确保compress_len是偶数
        if compress_len % 2 != 0:
            raise ValueError("compress_len必须是偶数")
            
        # 调用父类初始化，但压缩长度为compress_len的一半
        super().__init__(dimension, compress_len // 2, num_heads, dropout, weight_alpha)
        
        # 交叉注意力层的投影矩阵
        self.cross_q_proj = nn.Linear(dimension, dimension)
        self.cross_k_proj = nn.Linear(dimension, dimension)
        self.cross_v_proj = nn.Linear(dimension, dimension)
        
        # 输出投影
        self.cross_out_proj = nn.Linear(dimension, dimension)
        
        # Dropout
        self.cross_dropout = nn.Dropout(dropout)

    def forward(self, Fv, extern_weight=None, mask=None):
        """
        前向传播
        
        参数:
            Fv: 长期视频记忆特征，形状为 (batch_size, vid_len, dimension)
            extern_weight: 外部提供的权重，形状为 (batch_size, vid_len)
            mask: 有效位置的掩码，形状为 (batch_size, vid_len)
            
        返回:
            Fv_concat: 拼接后的视频特征，形状为 (batch_size, compress_len, dimension)
            compress_mask: 拼接后的掩码，形状为 (batch_size, compress_len)
        """
        batch_size, vid_len, dimension = Fv.size()
        device = Fv.device
        half_compress_len = self.compress_len  # 实际压缩长度是总长度的一半
        full_compress_len = half_compress_len * 2  # 最终长度
        
        # 1. 获取压缩的视频特征 Fs (前半部分的基础)
        Fs, compress_mask_half = super().forward(Fv, extern_weight, mask)
        
        # 2. 提取每个batch的最近compress_len/2个有效特征作为Fo
        bool_mask = mask.bool() if mask is not None else torch.ones(batch_size, vid_len, dtype=torch.bool, device=device)
        Fo_list = []
        actual_fo_lengths = []
        
        for batch in range(batch_size):
            # 获取当前批次的有效特征
            valid_features = Fv[batch][bool_mask[batch]]
            # 记录实际有效特征长度
            valid_len = valid_features.shape[0]
            actual_fo_lengths.append(min(valid_len, half_compress_len))
            # 取最后compress_len/2个
            Fo_list.append(valid_features[-half_compress_len:] if valid_len >= half_compress_len 
                          else valid_features)
        
        # 3. 创建Fo的填充张量
        padded_Fo = torch.zeros(batch_size, half_compress_len, dimension, device=device)
        fo_mask = torch.zeros(batch_size, half_compress_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            fo_len = Fo_list[i].shape[0]
            # 填充Fo
            if fo_len < half_compress_len:
                padded_Fo[i, :fo_len] = Fo_list[i]
                fo_mask[i, :fo_len] = True
            else:
                padded_Fo[i] = Fo_list[i]
                fo_mask[i] = torch.ones(half_compress_len, dtype=torch.bool, device=device)
        
        
        # 4. 创建结果变量，初始化为零张量
        Fv_concat = torch.zeros(batch_size, full_compress_len, dimension, device=device)
        full_compress_mask = torch.zeros(batch_size, full_compress_len, dtype=torch.bool, device=device)
        
        # 5. 处理特殊情况：有效特征总数小于full_compress_len
        special_cases = []
        for i in range(batch_size):
            # 获取当前批次的有效特征总数
            valid_features_count = bool_mask[i].sum().item()
            
            if valid_features_count < full_compress_len:
                # 特殊情况：有效特征总数小于full_compress_len
                # 直接使用原始的所有有效特征，pad到full_compress_len
                valid_features = Fv[i][bool_mask[i]]
                # 放入结果张量的前valid_features_count位置
                Fv_concat[i, :valid_features_count] = valid_features
                # 设置对应的mask
                full_compress_mask[i, :valid_features_count] = True
                # 记录特殊情况的batch索引
                special_cases.append(i)
        # 创建一个掩码，标记哪些batch需要进行交叉注意力计算
        batch_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for idx in special_cases:
            batch_mask[idx] = False

        # 6. 对非特殊情况的样本执行交叉注意力计算
        if batch_mask.any():
            # 筛选需要进行注意力计算的batch
            filtered_Fo = padded_Fo[batch_mask]
            filtered_Fs = Fs[batch_mask]
            filtered_fo_mask = fo_mask[batch_mask]
            
            # 交叉注意力计算 - Fo作为query，Fs作为key和value
            # 6.1 投影query, key, value
            q = self.cross_q_proj(filtered_Fo)  # (filtered_batch_size, half_compress_len, dimension)
            k = self.cross_k_proj(filtered_Fs)  # (filtered_batch_size, half_compress_len, dimension)
            v = self.cross_v_proj(filtered_Fs)  # (filtered_batch_size, half_compress_len, dimension)
            
            # 6.2 分割多头
            filtered_batch_size = filtered_Fo.size(0)
            head_dim = dimension // self.num_heads
            
            # 重塑为多头格式
            q = q.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(filtered_batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            
            # 6.3 计算注意力分数
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            
            # 应用Fo的掩码（确保填充部分不参与注意力计算）
            if filtered_fo_mask is not None:
                # 扩展掩码以适应多头注意力的形状
                q_mask = filtered_fo_mask.unsqueeze(1).unsqueeze(-1)
                # 在无效查询位置应用大的负值
                scores = scores.masked_fill(~q_mask.bool(), -1e9)
            
            # 应用softmax得到注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.cross_dropout(attn_weights)
            
            # 6.4 计算加权和
            weighted_v = torch.matmul(attn_weights, v)
            
            # 6.5 合并多头结果
            weighted_v = weighted_v.transpose(1, 2).contiguous()
            weighted_v = weighted_v.view(filtered_batch_size, half_compress_len, dimension)
            
            # 6.6 应用输出投影
            attn_output = self.cross_out_proj(weighted_v)
            
            # 6.7 使用交叉注意力的输出作为前半部分特征
            # 这里直接使用attn_output，而不是残差连接
            first_half_features = attn_output
            
            # 将结果放入最终拼接张量
            for idx, orig_idx in enumerate(torch.where(batch_mask)[0]):
                # 前半部分：通过交叉注意力处理的特征
                Fv_concat[orig_idx, :half_compress_len] = first_half_features[idx]
                full_compress_mask[orig_idx, :half_compress_len] = filtered_fo_mask[idx]
                
                # 后半部分：最近的half_compress_len个特征
                Fv_concat[orig_idx, half_compress_len:] = padded_Fo[orig_idx]
                full_compress_mask[orig_idx, half_compress_len:] = fo_mask[orig_idx]
        
        return Fv_concat, full_compress_mask