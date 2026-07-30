import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import time
import functools
from typing import Any, Callable

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

class InterMemory(nn.Module):
    """
    交互式记忆模块：用于管理历史信息并预测未来特征
    基于MGSL-Net的记忆机制，包含记忆槽的维护和基于记忆的特征预测
    """
    def __init__(self, 
                 d_model=256,           # 模型特征维度
                 future_length=32,      # 未来特征长度
                 memory_slots=64,       # 记忆槽数量
                 similarity_threshold=0.1, # 相似度阈值，控制记忆槽更新
                 num_heads=8):          # 注意力头数量
        super(InterMemory, self).__init__()
        
        self.d_model = d_model
        self.future_length = future_length
        self.memory_slots = memory_slots
        self.similarity_threshold = similarity_threshold
        self.num_heads = num_heads
        self.updateCnt = 0
        
        # 初始化记忆槽，形状为 [memory_slots, d_model]
        self.memory = nn.Parameter(torch.randn(memory_slots, d_model))
        # L2归一化
        # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
        
        # 特征映射层，用于生成记忆更新的写入键和写入值
        self.write_key_proj = nn.Linear(d_model, d_model)
        self.write_val_proj = nn.Linear(d_model, d_model)
        self.erase_val_proj = nn.Linear(d_model, d_model)
        
        # 未来特征初始化嵌入，指定设备
        self.future_embedding = nn.Parameter(torch.randn(future_length, d_model))
        
        # 用于从记忆中读取信息的注意力机制
        self.memory_read_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 用于对记忆读取结果进行自注意力处理的模块
        self.memory_self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 跟踪哪些视频帧已经被更新到记忆中
        self.updated_frames = defaultdict(set)

        # 记忆槽使用频率跟踪
        self.register_buffer('slot_usage_counter', torch.zeros(memory_slots))
        # Step1: 语义锚点（header token），用于文本驱动槽选择
        self.header_tokens = nn.Parameter(torch.randn(memory_slots, d_model))
        
    def reset(self):
        self.memory = nn.Parameter(torch.randn(self.memory_slots, self.d_model))
        self.updateCnt = 0
        self.slot_usage_counter.zero_()
        self.updated_frames = defaultdict(set)
        # L2归一化
        # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)

    def getUpdateCnt(self):
        return self.updateCnt
    
    def setInterMemory(self, memory, updateCnt):
        self.memory = memory
        self.updateCnt = updateCnt

    # 后续可以加一个set或者表，记录下哪些帧已经更新过了（qid, idx）作为键，只更新没有更新过的帧
    def update_memory(self, features, short_start, vid, mask=None):
        """
        基于输入特征更新记忆槽
        实现基于相似度的选择性写入和擦除机制
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            short_start: 每个批次样本在原视频中的起始帧索引，形状为 [batch_size]
            vid: 每个批次样本所属的视频ID，形状为 [batch_size]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
        """

        self.updateCnt += 1

        batch_size, seq_length, _ = features.shape

        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            # 确保mask是布尔类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 逐帧/逐特征处理输入
        for b in range(batch_size):
            # 计算当前批次的有效帧数量
            valid_feat = mask[b].sum().item()
            
            if valid_feat == 0:
                continue
            
            # 获取当前批次有效帧的索引（按照从左到右的顺序）
            valid_indices = torch.where(mask[b])[0]

            # for t in range(seq_length):
            for i in range(valid_feat-1, -1, -1):
                t = valid_indices[i]  # 当前处理的帧在seq_length中的位置
                
                # 计算原视频中的帧索引
                # 从右往左第i+1个有效帧对应视频中的(short_start-i-1)帧
                frame_idx = short_start[b].item() - (valid_feat - 1 - i)
                video_id = vid[b]
                
                # 检查该帧是否已经被更新到记忆中
                frame_key = (video_id, frame_idx)
                if frame_key in self.updated_frames:
                    # 如果已经更新过，则跳过
                    continue
                
                # 将该帧标记为已更新
                self.updated_frames[video_id].add(frame_idx)

                # 获取当前帧的特征
                feature = features[b, t].unsqueeze(0)  # [1, d_model]
                
                # 生成写入键
                write_key = self.write_key_proj(feature)  # [1, d_model]
                
                # 计算写入键与所有记忆槽的余弦相似度
                # 对记忆槽和写入键进行L2归一化
                memory_norm = F.normalize(self.memory, p=2, dim=1)
                key_norm = F.normalize(write_key, p=2, dim=1)
                
                # 计算余弦相似度 [memory_slots]
                similarity = torch.matmul(memory_norm, key_norm.t()).squeeze()
                
                # 使用softmax获取写入权重
                write_weights = F.softmax(similarity, dim=0)
                
                # 生成写入值和擦除值
                write_val = self.write_val_proj(feature)  # [1, d_model]
                erase_val = torch.sigmoid(self.erase_val_proj(feature))  # [1, d_model]，值范围(0,1)
                
                # 动态调整相似度阈值
                sim_thd = self.slot_usage_counter*self.similarity_threshold
                sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
                update_mask = (similarity > sim_thd)

                # # 固定阈值
                # update_mask = (similarity > self.similarity_threshold)
                
                if update_mask.any():
                    # 对需要更新的记忆槽应用擦除和写入操作
                    for i in range(self.memory_slots):
                        if update_mask[i]:
                            self.slot_usage_counter[i] += 1
                            # 基于相似度的擦除与写入（类似那篇文章中的更新公式）
                            # m_i' = m_i * (1 - w_i * e_t) + w_i * u_t
                            # 但是说实话我不太理解这里et是什么作用，一个从输入特征线性映射+sigmoid得到的值，为什么能指导旧记忆的保留程度
                            erase_term = 1.0 - write_weights[i] * erase_val
                            # 不用erase_val，更新退化为特征和旧记忆的加权融合
                            # erase_term = 1.0 - write_weights[i]
                            write_term = write_weights[i] * write_val
                            
                            # 更新对应的记忆槽
                            self.memory.data[i] = self.memory.data[i] * erase_term + write_term
        # 对更新后的记忆进行L2归一化
        # self.memory.data = F.normalize(self.memory.data, p=2, dim=-1)
        if torch.isnan(self.memory).any():
            print("WARNING: after update, memory contains NaN")

    def read_memory(self, query_features, mask=None):
        """
        从记忆中读取与查询特征相关的信息
        
        Args:
            query_features: 拼接特征，形状为 [batch_size, concat_seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, concat_seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
        Returns:
            memory_output: 从记忆中检索的输出，形状为 [batch_size, concat_seq_length, d_model]
        """
        batch_size = query_features.shape[0]
        
        # 扩展记忆以匹配批次大小
        # 形状: [batch_size, memory_slots, d_model]
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        if mask is None:
            attn_mask = None
        else:
            if mask.dtype == torch.bool:
                # 如果是布尔型，True=有效，False=填充
                mask_float = mask.float()  # True->1.0, False->0.0
            else:
                # 如果是数值型，假设 1=有效，0=填充
                mask_float = mask.clamp(0, 1)  # 确保是 0/1
            # 转换mask为float类型，并将填充位(False)变为-inf，有效位(True)变为0
            # [batch_size, concat_seq_length] -> [batch_size, concat_seq_length, 1]
            attn_mask = mask_float.unsqueeze(-1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            # 加性，有效位加0不影响，填充位加-inf，softmax后会趋近于0
            # 将attn_mask扩展到[batch_size, concat_seq_length, memory_slots]
            attn_mask = attn_mask.expand(-1, -1, self.memory_slots)
            # 调整为多头注意力需要的形状 [B*num_heads, L, M]
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)  # 复制到每个注意力头


        
        # 使用注意力机制从记忆中读取信息
        # Q: 查询特征 [batch_size, concat_seq_length, d_model]
        # K, V: 记忆 [batch_size, memory_slots, d_model]
        memory_output, _ = self.memory_read_attention(
            query=query_features,
            key=memory_expanded,
            value=memory_expanded,
            attn_mask=attn_mask
        )

        if torch.isnan(memory_output).any():
            print("WARNING: in interMemory, read_memory, memory_output contains NaN")
        
        return memory_output
    
    def read_memory_avg(self, query_features, mask=None):
        """
        从记忆中读取与查询特征相关的信息
        新方法：将每个槽独立处理，然后对结果求平均
        
        Args:
            query_features: 拼接特征，形状为 [batch_size, concat_seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, concat_seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
        Returns:
            memory_output: 从记忆中检索的输出，形状为 [batch_size, concat_seq_length, d_model]
        """
        batch_size, seq_length, _ = query_features.shape
        device = query_features.device
        
        # 创建一个存储所有slot注意力结果的张量
        all_slot_outputs = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        
        # 处理掩码
        if mask is None:
            attn_mask = None
        else:
            if mask.dtype == torch.bool:
                mask_float = mask.float()  # True->1.0, False->0.0
            else:
                mask_float = mask.clamp(0, 1)  # 确保是 0/1
            
            # 转换为注意力掩码格式
            attn_mask = mask_float.unsqueeze(-1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        
        # 遍历每个记忆槽
        for i in range(self.memory_slots):
            # 获取当前记忆槽
            # 形状：[d_model] -> [1, 1, d_model] -> [batch_size, 1, d_model]
            current_slot = self.memory[i].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            
            # 如果有掩码，调整为当前记忆槽的格式
            if attn_mask is not None:
                # 对于单个槽，只需要 [batch_size, seq_length, 1]
                current_mask = attn_mask[:, :, 0:1]
                # 调整为多头注意力需要的形状
                current_mask = current_mask.repeat_interleave(self.num_heads, dim=0)
            else:
                current_mask = None
            
            # 对当前槽执行注意力
            slot_output, _ = self.memory_read_attention(
                query=query_features,
                key=current_slot,
                value=current_slot,
                attn_mask=current_mask
            )
            
            # 将结果添加到输出中
            all_slot_outputs += slot_output
        
        # 计算平均值
        memory_output = all_slot_outputs / self.memory_slots
        
        if torch.isnan(memory_output).any():
            print("WARNING: in interMemory, read_memory_avg, memory_output contains NaN")
        
        return memory_output
    
    def predict_future(self, history_features, current_features, mask=None):
        """
        基于历史特征、当前特征和记忆预测未来特征
        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
        Returns:
            future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
        """
        batch_size = history_features.shape[0]
        
        # 扩展未来嵌入以匹配批次大小
        # 形状: [batch_size, future_length, d_model]
        future_embed = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接历史特征、当前特征和未来嵌入
        # 形状: [batch_size, (history_length + current_length + future_length), d_model]
        concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

        # 拼接特征的mask,形状为 [batch_size, (history_length + current_length + future_length)]
        if mask is not None:
            current_mask = torch.ones(batch_size, current_features.shape[1], 
                                device=history_features.device, dtype=mask.dtype)
            future_mask = torch.ones(batch_size, future_embed.shape[1], 
                               device=history_features.device, dtype=mask.dtype)
            mask = torch.cat([mask, current_mask, future_mask], dim=1)

        # 从记忆中读取相关信息
        # memory_output = self.read_memory(concatenated_features, mask)
        memory_output = self.read_memory_avg(concatenated_features, mask)

        # 用读取了记忆信息的结果做self-attention，使未来嵌入部分能够关注下历史帧和当前帧的信息
        memory_output = self.memory_self_attention(memory_output, memory_output, memory_output)[0]
        
        # 只提取增强特征中的未来部分（最后future_length个时间步）
        # 形状: [batch_size, future_length, d_model]
        future_part = memory_output[:, -self.future_length:, :]
        
        if torch.isnan(future_part).any():
            print("WARNING: in interMemory, predict_future, future_part contains NaN")

        # 返回一个全为1，形状为 [batch_size, future_length]的mask
        future_mask = torch.ones(batch_size, self.future_length, 
                                device=history_features.device, 
                                dtype=mask.dtype if mask is not None else torch.bool)
        
        return future_part, future_mask



class InterMemorySeq(nn.Module):
    """
    序列化interMemory，每个记忆槽存储特征序列而不是一个特征向量
    支持自动优化记忆槽多样性的版本
    """
    def __init__(self, 
                 d_model=256,            # 模型特征维度
                 future_length=32,       # 未来特征长度
                 memory_slots=64,        # 记忆槽数量
                 slot_len=64,            # 每个槽的序列长度
                 compress_len=10,        # 压缩后历史帧长度 T_c，Step2 滑窗宽度
                 similarity_threshold=0.1, # 相似度阈值，控制记忆槽更新
                 num_heads=8,            # 注意力头数量
                 pooling_type='mean',    # 池化类型：'mean'或'max'
                 # 新增参数用于自动优化
                 diversity_loss_weight=0.5,  # 多样性损失权重
                 optimization_interval=10,   # 优化间隔（多少次forward后进行一次优化）
                 optimizer_lr=1e-4,         # 内部优化器学习率
                 topk=4,                    
                 ema_lambda=0.05,        # Step4 EMA 更新系数
                 slot_select_mode="header",
                 disable_phase_alignment=False,
                 fixed_phase_start=0,
                 legacy_v1=False,
                 ):
        super(InterMemorySeq, self).__init__()

        # 原有参数
        self.d_model = d_model
        self.legacy_v1 = legacy_v1
        self.future_length = future_length
        self.memory_slots = memory_slots
        self.slot_len = slot_len
        self.compress_len = compress_len
        self.similarity_threshold = similarity_threshold
        self.num_heads = num_heads
        self.pooling_type = pooling_type
        self.updateCnt = 0
        self.topk = topk
        self.ema_lambda = ema_lambda
        self.slot_select_mode = slot_select_mode
        self.disable_phase_alignment = disable_phase_alignment
        self.fixed_phase_start = int(fixed_phase_start)

        # 新增自动优化相关参数
        self.diversity_loss_weight = diversity_loss_weight
        self.optimization_interval = optimization_interval
        
        # 初始化序列化记忆槽，形状为 [memory_slots, slot_len, d_model]

        # 1.随机初始化
        self.memory = nn.Parameter(torch.randn(memory_slots, slot_len, d_model))
        # 2.正交初始化
        # memory = torch.empty(memory_slots, slot_len, d_model)
        # nn.init.orthogonal_(memory)
        # self.memory = nn.Parameter(memory)
        # 3.傅里叶基初始化
        # def fourier_basis_initialization(memory_slots, slot_len, d_model):
        #     """用不同频率的傅里叶基函数初始化记忆槽"""
        #     memory = torch.zeros(memory_slots, slot_len, d_model)
            
        #     # 生成不同的频率模式
        #     freqs = torch.linspace(0.1, 1.0, memory_slots)
            
        #     for i in range(memory_slots):
        #         for j in range(d_model):
        #             # 每个特征维度用不同相位的正弦波
        #             phase = j * 2 * torch.pi / d_model
        #             memory[i, :, j] = torch.sin(freqs[i] * torch.arange(slot_len) + phase)
            
        #     return memory

        # self.memory = nn.Parameter(fourier_basis_initialization(memory_slots, slot_len, d_model))

        # 记忆槽使用频率跟踪
        self.register_buffer('slot_usage_counter', torch.zeros(memory_slots))

        # Step1: 语义锚点；legacy_v1 ckpt 不含 header_tokens
        if not self.legacy_v1:
            self.header_tokens = nn.Parameter(torch.randn(memory_slots, d_model))
        
        # 优化步数计数器
        self.register_buffer('forward_counter', torch.tensor(0))

        # 特征映射层，用于相似度计算
        self.query_proj = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        
        # 交叉注意力层，用于记忆更新
        self.memory_update_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 控制门机制，决定更新比例
        self.update_gate = nn.Linear(d_model * 2, d_model)
        
        # 未来特征初始化嵌入
        self.future_embedding = nn.Parameter(torch.randn(future_length, d_model))
        self.future_proj = nn.Linear(d_model, d_model)
        
        # read操作用到的注意力机制，在老版predict中用于和slots跨膜太交互，新版的只用和fused memory交互
        self.memory_read_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        current_length = 8  # present的采样长度，这个参数只在这里用到了，先这样用着，另外历史帧长度用的slot_len，这点需要在InterMemory类定义的时候确保一致
        # predict操作中，函数predict_future_direct_crossModal_andCompress中用到的压缩层
        self.fc_compressor = nn.Linear(slot_len + current_length + future_length, future_length)

        # 定义FC预测头
        self.fc_prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 用于对记忆读取结果进行自注意力处理的模块
        self.memory_self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 简单分类器 - MLP，用于判断输入特征是否更新
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

        # Step3 子模块仅在新版 4-step 路径启用
        if not self.legacy_v1:
            self.step3_cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                batch_first=True,
            )
            self.step3_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        # 内部优化器，用于优化记忆多样性
        self.memory_optimizer = torch.optim.Adam([self.memory], lr=optimizer_lr)
        # 缓存最近一次 Step4 的结构约束损失，供外部训练脚本读取
        self.last_memory_struct_loss = torch.tensor(0.0)
            

    def reset(self):
        """重置记忆状态、header 语义锚点和计数器"""
        self.memory.data = torch.randn(self.memory_slots, self.slot_len, self.d_model)
        if not self.legacy_v1:
            self.header_tokens.data = torch.randn(self.memory_slots, self.d_model)
        self.slot_usage_counter.zero_()
        self.forward_counter.zero_()
        self.updateCnt = 0

    def getUpdateCnt(self):
        return self.updateCnt
    
    def setInterMemory(self, memory=None,future_embedding=None, updateCnt=None):
        if memory is not None:
            self.memory = memory
        if future_embedding is not None:
            self.future_embedding = future_embedding
        if updateCnt is not None:
            self.updateCnt = updateCnt

    def classify(self, x):
        """
        分类方法，直接返回0/1结果
        推理的时候调用，可以在之前加个 with torch.no_grad():

        Args:
            x: [batch_size, seq_length, d_model]
        
        Returns:
            predictions: [batch_size] 0/1分类结果
        """
        pooled = torch.mean(x, dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return (torch.sigmoid(logits) > 0.5).long()

    def compute_diversity_loss_negCos(self):
        """
        计算记忆槽多样性损失
        使用负余弦相似度鼓励记忆槽之间的差异化
        
        Returns:
            diversity_loss: 多样性损失值
        """

        device = self.memory.device
        frob_similarity_matrix = torch.zeros(self.memory_slots, self.memory_slots, device=device)

        # 将self.memory投影
        projected_memory = self.memory_proj(self.memory)

        # 计算所有槽对之间的相似度
        for i in range(self.memory_slots):
            for j in range(i, self.memory_slots):  # 只计算上三角，利用对称性
                if i == j:
                    frob_similarity_matrix[i, j] = 1.0  # 自相似度为1
                else:
                    sim = self.frobenius_similarity(projected_memory[i], projected_memory[j])
                    # sim = self.hybrid_pool_similarity(projected_memory[i], projected_memory[j])
                    # sim = self.multi_scale_semantic_similarity(projected_memory[i], projected_memory[j])
                    frob_similarity_matrix[i, j] = sim
                    frob_similarity_matrix[j, i] = sim  # 对称赋值
        
        # 排除对角线元素（自己与自己的相似度）
        mask = torch.eye(self.memory_slots, device=self.memory.device, dtype=torch.bool)
        frob_similarity_matrix = frob_similarity_matrix.masked_fill(mask, 0.0)
        
        # 计算平均余弦相似度作为损失（希望这个值越小越好）
        # 分母是 N(N-1)，即除了对角线的所有元素数量
        diversity_loss = frob_similarity_matrix.sum() / (self.memory_slots * (self.memory_slots - 1))
        
        return diversity_loss

    def compute_diversity_loss_ortho(self):
        """
        计算基于正交化正则的记忆槽多样性损失
        使用Frobenius范数计算MM^T与单位矩阵的差异
        这个loss的范围应该在[0,2N(N−1)]，N是记忆槽的数量
        
        Returns:
            ortho_loss: 正交性损失值 (越小表示记忆槽越正交)
        """
        # 获取记忆槽的表示 (memory_slots, slot_len, d_model)
        memory = self.memory
        
        # 沿序列维度做平均池化，得到每个槽的聚合表示 (memory_slots, d_model)
        memory_repr = memory.mean(dim=1)
        
        # 对记忆槽表示进行L2归一化 
        memory_repr = F.normalize(memory_repr, p=2, dim=1)
        
        # 计算MM^T (memory_slots, memory_slots)
        mmt = torch.mm(memory_repr, memory_repr.t())
        
        # 创建单位矩阵
        identity = torch.eye(self.memory_slots, device=memory.device)
        
        # 计算Frobenius范数的平方差
        ortho_loss = torch.norm(mmt - identity, p='fro') ** 2
        
        return ortho_loss

    # @countTime
    def optimize_memory_diversity(self):
        """
        执行记忆多样性优化
        """
        
        with torch.set_grad_enabled(True):  # 临时启用梯度
            # 计算多样性损失
            # diversity_loss = self.compute_diversity_loss_negCos()
            diversity_loss = self.compute_diversity_loss_ortho()
            
            # 加权损失
            memory_loss = self.diversity_loss_weight * diversity_loss
            
            # 执行优化步骤
            self.memory_optimizer.zero_grad()
            try:
                memory_loss.backward()
            except Exception as e:
                print(f"WARNING: Memory diversity optimization failed: {e}")
                return
            self.memory_optimizer.step()
        
        with open("interMemory_Optimization.log", 'a') as f:
            f.write(f"Memory diversity optimization: loss = {memory_loss.item():.6f}\n")
        print(f"Memory diversity optimization: loss = {memory_loss.item():.6f}")

    def frobenius_similarity(self, seq1, seq2, mask1=None, mask2=None):
        """
        计算两个带掩码的序列的Frobenius相似度
        Args:
            seq1: [seq_len1, d_model]
            seq2: [seq_len2, d_model]
            mask1: [seq_len1], 1表示有效，0表示无效（可选）
            mask2: [seq_len2], 1表示有效，0表示无效（可选）
        Returns:
            similarity: 标量值
        """
        # 归一化序列
        seq1_norm = F.normalize(seq1, p=2, dim=-1)
        seq2_norm = F.normalize(seq2, p=2, dim=-1)

        # 应用掩码（如果提供）
        if mask1 is not None:
            seq1_norm = seq1_norm * mask1.unsqueeze(-1).float()
        if mask2 is not None:
            seq2_norm = seq2_norm * mask2.unsqueeze(-1).float()
        
        sim_matrix = torch.matmul(seq1_norm, seq2_norm.t())  # [seq_len1, seq_len2]

        # 计算有效元素数量
        if mask1 is not None and mask2 is not None:
            valid_counts = mask1.float().sum() * mask2.float().sum()
        elif mask1 is not None:
            valid_counts = mask1.float().sum() * seq2.shape[0]
        elif mask2 is not None:
            valid_counts = seq1.shape[0] * mask2.float().sum()
        else:
            valid_counts = seq1.shape[0] * seq2.shape[0]

        # 平均相似度
        similarity = sim_matrix.sum() / (valid_counts + 1e-8)
        return similarity

    def temporal_pool(self, x, pool_layers=2, mask=None):
        """
        时序感知层次化池化
        Args:
            x: [seq_len, d_model] 输入序列，seq_len是序列长度，d_model是特征维度
            mask: [seq_len] (可选) 用于处理变长序列的掩码
        Returns:
            池化后的特征向量 [d_model]
        """
        # 如果提供了掩码，应用掩码
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            
        # 多尺度时序池化
        # 调整维度为 [1, d_model, seq_len] 以适应1D卷积
        current_seq = x.permute(1, 0).unsqueeze(0)  
        
        # 进行多层池化
        for _ in range(pool_layers):
            # 混合池化层 - 结合平均池化和最大池化的优势
            avg_pool = F.avg_pool1d(current_seq, kernel_size=3, stride=2, padding=1)
            max_pool = F.max_pool1d(current_seq, kernel_size=3, stride=2, padding=1)
            current_seq = 0.5 * (avg_pool + max_pool)  # 平均池化和最大池化的加权平均
            
        # 最终聚合 - 结合全局平均和最大池化
        return 0.5 * (current_seq.mean(dim=-1) + current_seq.max(dim=-1)[0])

    # 计算混合池化相似度
    def hybrid_pool_similarity(self, seq1, seq2, mask1=None, mask2=None):
        """
        计算两个视频序列的相似度
        Args:
            seq1: [seq_len1, d_model] 第一个视频序列
            seq2: [seq_len2, d_model] 第二个视频序列
            mask1: [seq_len1] (可选) 第一个序列的掩码
            mask2: [seq_len2] (可选) 第二个序列的掩码
        Returns:
            相似度标量 (范围[-1,1])
        """
        # 归一化处理 - 使特征向量位于单位球面上
        seq1 = F.normalize(seq1, p=2, dim=-1)
        seq2 = F.normalize(seq2, p=2, dim=-1)
        
        # 时序特征提取
        feat1 = self.temporal_pool(seq1, pool_layers=2, mask=mask1).squeeze()
        feat2 = self.temporal_pool(seq2, pool_layers=2, mask=mask2).squeeze()
        
        # 相似度计算 - 使用余弦相似度
        return F.cosine_similarity(feat1, feat2, dim=0)

    # 计算多尺度相似度
    def multi_scale_semantic_similarity(self, seq1, seq2, mask1=None, mask2=None, 
                                   n_segments=3, top_k_frames=5, 
                                   weights=(0.4, 0.3, 0.3)):
        """
        多尺度语义相似度计算 - 结合全局、分段、关键帧三个层次
        
        Args:
            seq1: [seq_len1, d_model] - 第一个视频序列的特征
            seq2: [seq_len2, d_model] - 第二个视频序列的特征  
            mask1: [seq_len1] - 第一个序列的掩码，1表示有效，0表示无效（可选）
            mask2: [seq_len2] - 第二个序列的掩码，1表示有效，0表示无效（可选）
            n_segments: int - 时间分段数量，默认3段（开始、中间、结束）
            top_k_frames: int - 提取的关键帧数量，默认5帧
            weights: tuple - 三个层次的权重 (全局, 分段, 关键帧)，默认(0.4, 0.3, 0.3)
            
        Returns:
            similarity: torch.Tensor - 标量，语义相似度分数 [0, 1]
        """
        
        # ==================== 数据预处理 ====================
        
        # 如果其中一个有效位长度为0，直接返回0
        if mask1 is not None and mask1.sum().item() == 0:
            return 0.0
        if mask2 is not None and mask2.sum().item() == 0:
            return 0.0

        # L2归一化，便于余弦相似度计算
        seq1_norm = F.normalize(seq1, p=2, dim=-1, eps=1e-8)
        seq2_norm = F.normalize(seq2, p=2, dim=-1, eps=1e-8)
        
        # 应用掩码，将无效位置的特征置零
        if mask1 is not None:
            seq1_norm = seq1_norm * mask1.unsqueeze(-1).float()
            valid_len1 = mask1.sum().item()  # 有效帧数
        else:
            valid_len1 = seq1.shape[0]
            
        if mask2 is not None:
            seq2_norm = seq2_norm * mask2.unsqueeze(-1).float()
            valid_len2 = mask2.sum().item()  # 有效帧数
        else:
            valid_len2 = seq2.shape[0]
        
        # 提取有效序列（去除padding部分）
        seq1_valid = seq1_norm[:valid_len1]
        seq2_valid = seq2_norm[:valid_len2]
        
        
        # ==================== 层次1: 全局语义相似度 ====================
        
        # 计算整个视频的全局语义表示（平均池化）
        global_feat1 = seq1_valid.mean(dim=0)  # [d_model]
        global_feat2 = seq2_valid.mean(dim=0)  # [d_model]
        
        # 计算全局语义相似度
        global_similarity = F.cosine_similarity(global_feat1, global_feat2, dim=0)
        
        # ==================== 层次2: 时间分段语义相似度 ====================
        
        def extract_temporal_segments(seq, n_segments):
            """
            将视频序列按时间均匀分段，提取每段的语义特征
            捕捉视频不同阶段的语义信息（如动作的开始、发展、结束）
            
            Args:
                seq: [seq_len, d_model] - 输入序列
                n_segments: int - 分段数量
                
            Returns:
                segments: [n_segments, d_model] - 每段的语义特征
            """
            seq_len = len(seq)
            seg_len = seq_len // n_segments  # 每段的长度
            segments = []
            
            for i in range(n_segments):
                # 计算当前段的起始和结束位置
                start_idx = i * seg_len
                end_idx = start_idx + seg_len if i < n_segments - 1 else seq_len
                
                # 提取当前段的特征（确保不超出序列长度）
                if start_idx < seq_len:
                    # 段内平均
                    segment = seq[start_idx:end_idx]
                    segment_feat = segment.mean(dim=0) if len(segment) > 0 else torch.zeros_like(seq[0])
                    segments.append(segment_feat)
            
            # 如果某些段为空，用零向量填充
            while len(segments) < n_segments:
                segments.append(torch.zeros_like(seq[0]))
                
            return torch.stack(segments)
        
        # 提取两个视频的时间分段特征
        segments1 = extract_temporal_segments(seq1_valid, n_segments)  # [n_segments, d_model]
        segments2 = extract_temporal_segments(seq2_valid, n_segments)  # [n_segments, d_model]
        
        # 计算分段间的相似度矩阵
        # segment_sim_matrix[i,j] 表示视频1的第i段与视频2的第j段的相似度
        segment_sim_matrix = torch.matmul(segments1, segments2.t())  # [n_segments, n_segments]
        
        # 对于每个段，找到其在另一个视频中的最佳匹配段
        # 这允许不同视频中相同语义内容出现在不同时间位置
        best_matches = segment_sim_matrix.max(dim=1)[0]  # [n_segments]
        segment_similarity = best_matches.max()  # 所有段的最佳匹配相似度，不用mean是因为只要有分成匹配的两个段，后续update会通过attn自动对应上
        
        # ==================== 层次3: 关键帧语义相似度 ====================
        
        def extract_key_frames(seq, top_k):
            """
            提取视频中最重要的关键帧
            使用特征范数作为重要性指标，范数大的帧通常包含更丰富的语义信息
            
            Args:
                seq: [seq_len, d_model] - 输入序列
                top_k: int - 提取的关键帧数量
                
            Returns:
                key_features: [min(top_k, seq_len), d_model] - 关键帧特征
            """
            # 计算每帧特征的L2范数作为重要性分数
            # 范数越大，表示该帧的特征越"突出"，语义信息越丰富
            importance_scores = seq.norm(dim=1)  # [seq_len]
            
            # 选择top-k个最重要的帧
            actual_k = min(top_k, len(seq))
            top_indices = torch.topk(importance_scores, k=actual_k, dim=0)[1]
            
            # 提取关键帧特征
            key_features = seq[top_indices]  # [actual_k, d_model]
            
            return key_features
        
        # 提取两个视频的关键帧
        key_frames1 = extract_key_frames(seq1_valid, top_k_frames)
        key_frames2 = extract_key_frames(seq2_valid, top_k_frames)
        
        # 计算关键帧的整体语义表示（平均池化）
        key_feat1 = key_frames1.mean(dim=0)  # [d_model]
        key_feat2 = key_frames2.mean(dim=0)  # [d_model]
        
        # 计算关键帧语义相似度
        key_similarity = F.cosine_similarity(key_feat1, key_feat2, dim=0)
        
        # ==================== 多尺度融合 ====================
        
        # 将三个层次的相似度按权重融合
        # global: 捕捉整体语义主题
        # segment: 捕捉时间演变模式  
        # key: 捕捉关键语义事件
        final_similarity = (weights[0] * global_similarity + 
                        weights[1] * segment_similarity + 
                        weights[2] * key_similarity)
        
        # 确保结果在合理范围内
        final_similarity = torch.clamp(final_similarity, -1.0, 1.0)
        
        return final_similarity

    def compute_sequence_similarity_pool_cosSim(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            # 确保mask是布尔类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 根据指定的池化类型进行池化操作
        if self.pooling_type == 'mean':
            # 对特征进行池化，考虑mask
            mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_length, 1]
            pooled_features = (projected_features * mask_expanded).sum(dim=1)  # [batch_size, d_model]
            valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch_size, 1]
            pooled_features = pooled_features / valid_counts  # [batch_size, d_model]
        elif self.pooling_type == 'max':
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_length, 1]
            masked_features = projected_features.clone()
            masked_features[~mask_expanded] = -1e9
            pooled_features = torch.max(masked_features, dim=1)[0]  # [batch_size, d_model]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        # 对记忆槽进行池化
        if self.pooling_type == 'mean':
            pooled_memory = projected_memory.mean(dim=1)  # [memory_slots, d_model]
        elif self.pooling_type == 'max':
            pooled_memory = torch.max(projected_memory, dim=1)[0]  # [memory_slots, d_model]
        
        # 归一化池化后的特征和记忆，用于计算余弦相似度
        pooled_features_norm = F.normalize(pooled_features, p=2, dim=1)  # [batch_size, d_model]
        pooled_memory_norm = F.normalize(pooled_memory, p=2, dim=1)  # [memory_slots, d_model]
        
        # 计算余弦相似度
        similarities = torch.matmul(pooled_features_norm, pooled_memory_norm.t())
        
        return similarities

    @countTime
    def compute_sequence_similarity_frobenius(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的Frobenius相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 初始化相似度矩阵
        similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
        # 对每个batch和每个memory slot计算Frobenius相似度
        for b in range(batch_size):
            for m in range(memory_slots):
                current_features = projected_features[b]  # [seq_length, d_model]
                current_mask = mask[b] if mask is not None else None  # [seq_length]
                current_memory = projected_memory[m]  # [slot_len, d_model]
                
                similarities[b, m] = self.frobenius_similarity(
                    current_features, 
                    current_memory,
                    current_mask,
                    None
                )
        
        return similarities

    def compute_sequence_similarity_frobenius_optimized(self, features, mask=None):
        """
        优化版本：使用矢量化操作计算输入特征序列与所有记忆槽的Frobenius相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """

        # slow_simlirities = self.compute_sequence_similarity_frobenius(features, mask)
        # slow_simlirities1 = self.compute_sequence_similarity_frobenius(features, mask)
        # print("show 2 times call compute_sequence_similarity_frobenius  is the same:", torch.allclose(slow_simlirities, slow_simlirities1)) # rtol=1e-8, atol=1e-10
        # print("show 2 times call compute_sequence_similarity_frobenius  is abs the same:", slow_simlirities == slow_simlirities1)

        batch_size, seq_length, d_model = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 处理mask
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 归一化
        features_norm = F.normalize(projected_features, p=2, dim=-1)  # [batch_size, seq_length, d_model]
        memory_norm = F.normalize(projected_memory, p=2, dim=-1)     # [memory_slots, slot_len, d_model]
        
        # 应用mask到features
        mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_length, 1]
        features_masked = features_norm * mask_expanded  # [batch_size, seq_length, d_model]

        # 计算相似度矩阵 - 矢量化版本
        # 直接计算所有batch和memory slot之间的相似度
        sim_totals = torch.einsum('bsd,mtd->bsmt', features_masked, memory_norm)  # [batch_size, seq_length, memory_slots, slot_len]
        sim_sums = sim_totals.sum(dim=(1, 3))  # [batch_size, memory_slots]
        
        # 计算有效元素数量
        valid_counts_features = mask.float().sum(dim=1, keepdim=True)  # [batch_size, 1]
        valid_counts_memory = torch.full((memory_slots,), slot_len, device=features.device, dtype=torch.float)  # [memory_slots]
        
        # 广播计算总的有效元素数
        total_valid_counts = valid_counts_features * valid_counts_memory.unsqueeze(0)  # [batch_size, memory_slots]
        
        # 计算平均相似度
        similarities = sim_sums / (total_valid_counts + 1e-8)
        
        return similarities


    def compute_sequence_similarity_hybrid_pool(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的混合池化相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 初始化相似度矩阵
        similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
        # 对每个batch和每个memory slot计算Frobenius相似度
        for b in range(batch_size):
            for m in range(memory_slots):
                current_features = projected_features[b]  # [seq_length, d_model]
                current_mask = mask[b] if mask is not None else None  # [seq_length]
                current_memory = projected_memory[m]  # [slot_len, d_model]
                
                similarities[b, m] = self.hybrid_pool_similarity(
                    current_features, 
                    current_memory,
                    current_mask,
                    None
                )
        
        return similarities

    def compute_sequence_similarity_multi_scale(self, features, mask=None):
        """
        计算输入特征序列与所有记忆槽的多尺度相似度
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
                    
        Returns:
            similarities: 相似度矩阵，形状为 [batch_size, memory_slots]
        """
        batch_size, seq_length, _ = features.shape
        memory_slots, slot_len, _ = self.memory.shape
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=features.device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 将特征和记忆投影到相似度空间
        projected_features = self.query_proj(features)  # [batch_size, seq_length, d_model]
        projected_memory = self.memory_proj(self.memory)  # [memory_slots, slot_len, d_model]
        
        # 初始化相似度矩阵
        similarities = torch.zeros(batch_size, memory_slots, device=features.device)
        
        # 对每个batch和每个memory slot计算相似度
        for b in range(batch_size):
            for m in range(memory_slots):
                current_features = projected_features[b]  # [seq_length, d_model]
                current_mask = mask[b] if mask is not None else None  # [seq_length]
                current_memory = projected_memory[m]  # [slot_len, d_model]
                
                similarities[b, m] = self.multi_scale_semantic_similarity(
                    current_features, 
                    current_memory,
                    current_mask,
                    None,
                    n_segments=3,
                    top_k_frames=5,
                    weights=(0.4, 0.3, 0.3)
                )
        
        return similarities
        

    def update_memory_normal(self, features, mask=None, short_start=None, vid=None):
        """
        基于输入特征序列更新记忆槽
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前批次是否有有效特征
            if not mask[b].any():
                continue
            
            # 获取当前批次的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度超过阈值的记忆槽
            similarity = similarities[b]  # [memory_slots]
            
            # # 动态相似度阈值
            sim_thd = self.slot_usage_counter*self.similarity_threshold
            sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
            update_mask = (similarity > sim_thd)

            # 固定相似度阈值
            # update_mask = (similarity > self.similarity_threshold)
            
            if not update_mask.any():
                continue
            
            # 获取需要更新的记忆槽索引
            update_indices = torch.where(update_mask)[0]
            update_per_sample[b] = len(update_indices)
            
            # 更新所选记忆槽
            for idx in update_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)
        
        # 如果不是全0，打印每个样本更新的记忆槽数量
        if update_per_sample.sum() > 0:
            print("Update per sample:", update_per_sample)

        # # 打印相似度最大值和平均值
        # print("Similarity max:", similarities.max())
        # print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

    @countTime
    def update_memory_normal_topk(self, features, mask=None, short_start=None, vid=None):
        """
        基于输入特征序列更新记忆槽, 选择相似度最高的k个进行更新
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius_optimized(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前批次是否有有效特征
            if not mask[b].any():
                continue
            
            # 获取当前批次的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度最高的k个记忆槽
            similarity = similarities[b]  # [memory_slots]
            # 找出相似度最高的k个记忆槽
            topk_indices = torch.topk(similarity, k=self.topk, largest=True).indices
            
            # 更新所选记忆槽
            for idx in topk_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)

        # # 打印相似度最大值和平均值
        # print("Similarity max:", similarities.max())
        # print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

    def update_memory_normal_guiyi(self, features, mask=None, short_start=None, vid=None):
        """
        基于输入特征序列更新记忆槽,将相似度归一化后，根据相似度阈值更新
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前批次是否有有效特征
            if not mask[b].any():
                continue
            
            # 获取当前批次的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度超过阈值的记忆槽
            similarity = similarities[b]  # [memory_slots]
            # 相似度归一化到0-1区间
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
            
            # # 动态相似度阈值
            # sim_thd = self.slot_usage_counter*self.similarity_threshold
            # sim_thd = sim_thd.clamp(max=self.similarity_threshold*10)
            # update_mask = (similarity > sim_thd)

            # 固定相似度阈值
            update_mask = (similarity > self.similarity_threshold)
            
            if not update_mask.any():
                continue
            
            # 获取需要更新的记忆槽索引
            update_indices = torch.where(update_mask)[0]
            update_per_sample[b] = len(update_indices)
            
            # 更新所选记忆槽
            for idx in update_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)
        
        # 如果不是全0，打印更新的最大记忆槽数量，最小记忆槽数量，平均记忆槽数量
        if update_per_sample.sum() > 0:
            print("batch Update max:", update_per_sample.max())
            print("batch Update min:", update_per_sample.min())
            print("batch Update mean:", update_per_sample.mean())

        # 打印相似度最大值和平均值
        print("Similarity max:", similarities.max())
        print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

    def update_memory_normal_gtframes(self, features, mask=None, short_start=None, vid=None):
        """
        根据历史特征中和gt相交的帧数量/分类器输出决定是否更新
        更新的记忆槽仍基于相似度（topk）
        使用注意力机制融合信息到相似度高的记忆槽
        
        Args:
            features: 输入特征，形状为 [batch_size, seq_length, d_model]
            gt_frames_in_long_memory: 历史帧与GT窗口相交的帧数量，形状为[batch_size],如果不用分类器决定样本是否更更新则需要这个参数
            mask: 特征有效性掩码，形状为 [batch_size, seq_length]，1表示有效，0表示无效
                    如果为None，则视所有特征为有效
            short_start: 
            vid: 这两个暂时用不到了，先留着
        """
        self.updateCnt += 1
        batch_size, seq_length, _ = features.shape
        device = features.device
        
        # 如果没有提供mask，则默认所有特征都有效
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                mask = mask.bool()
        
        # 计算输入特征与记忆槽的相似度
        similarities = self.compute_sequence_similarity_frobenius(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_hybrid_pool(features, mask)  # [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_multi_scale(features, mask)  # [batch_size, memory_slots]

        # 统计每个样本输入，多少个记忆槽进行了更新
        update_per_sample = torch.zeros(batch_size, device=device, dtype=torch.int32)

        # 用MLP分类器判断这批中每个样本是否要进行更新
        # 平均池化: [batch_size, seq_length, d_model] -> [batch_size, d_model]
        avg_pooled_features = torch.mean(features, dim=1)
        # 分类: [batch_size, d_model] -> [batch_size]
        classifier_logits = self.classifier(avg_pooled_features).squeeze(-1)
        probs = torch.sigmoid(classifier_logits)
        update_decision = (probs > 0.5).long()

        # 逐批次处理更新
        for b in range(batch_size):
            # 检查当前样本是否有有效特征
            if not mask[b].any():
                continue
                
            # # 判断当前样本是否需要更新,如果当前样本的相交帧数量小于7（中位数），则不更新
            # if gt_frames_in_long_memory[b] < 7:
            #     continue

            # 根据分类器的输出判断是否需要更新
            if update_decision[b] == 0:
                continue
                
            
            # 获取当前样本的特征和掩码
            batch_features = features[b].unsqueeze(0)  # [1, seq_length, d_model]
            batch_mask = mask[b].unsqueeze(0)  # [1, seq_length]
            
            # 找出相似度最高的k个记忆槽
            similarity = similarities[b]  # [memory_slots]
            # 找出相似度最高的k个记忆槽
            topk_indices = torch.topk(similarity, k=self.topk, largest=True).indices
            
            # 更新所选记忆槽
            for idx in topk_indices:
                # 增加使用计数
                self.slot_usage_counter[idx] += 1
                
                # 获取当前记忆槽
                current_memory = self.memory[idx].unsqueeze(0)  # [1, slot_len, d_model]
                
                # 准备注意力掩码
                if not batch_mask.all():
                    attn_mask = batch_mask.float()
                    attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
                    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.slot_len, -1)
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                else:
                    attn_mask = None
                
                # 使用交叉注意力机制更新记忆
                updated_memory, _ = self.memory_update_attention(
                    query=current_memory,        # [1, slot_len, d_model]
                    key=batch_features,          # [1, seq_length, d_model]
                    value=batch_features,        # [1, seq_length, d_model]
                    attn_mask=attn_mask          # [num_heads, slot_len, seq_length] or None
                )
                
                # 使用门控机制控制更新比例
                concat_memory = torch.cat([current_memory, updated_memory], dim=-1)  # [1, slot_len, 2*d_model]
                update_gate = torch.sigmoid(self.update_gate(concat_memory))  # [1, slot_len, d_model]
                
                # 应用门控更新
                new_memory = current_memory * (1 - update_gate) + updated_memory * update_gate
                
                # 更新记忆槽
                self.memory.data[idx] = new_memory.squeeze(0)

        # 打印相似度最大值和平均值
        # print("Similarity max:", similarities.max())
        # print("Similarity mean:", similarities.mean())

        # 检查是否存在NaN值
        if torch.isnan(self.memory).any():
            print("Warning: Memory contains NaN after update")

        return classifier_logits

    def read_memory(self, query_features, mask=None):
        """
        从所有记忆槽中读取信息并平均结果
        
        Args:
            query_features: 查询特征，形状为 [batch_size, seq_length, d_model]
            mask: 查询特征的有效性掩码，形状为 [batch_size, seq_length]
            
        Returns:
            memory_output: 从记忆中读取的输出，形状为 [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = query_features.shape
        device = query_features.device
        
        # 创建一个存储所有槽注意力结果的张量
        all_slot_outputs = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        
        # 处理掩码
        if mask is None:
            attn_mask = None
        else:
            if mask.dtype == torch.bool:
                mask_float = mask.float()
            else:
                mask_float = mask.clamp(0, 1)
            
            attn_mask = mask_float.unsqueeze(-1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e9'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        
        # 遍历每个记忆槽
        for i in range(self.memory_slots):
            current_slot = self.memory[i].unsqueeze(0).expand(batch_size, -1, -1)
            
            if attn_mask is not None:
                current_mask = attn_mask.expand(-1, -1, self.slot_len)
                current_mask = current_mask.repeat_interleave(self.num_heads, dim=0)
            else:
                current_mask = None
            
            # 执行交叉注意力
            slot_output, _ = self.memory_read_attention(
                query=query_features,           # [batch_size, seq_length, d_model]
                key=current_slot,               # [batch_size, slot_len, d_model]
                value=current_slot,             # [batch_size, slot_len, d_model]
                attn_mask=current_mask          # [B*num_heads, seq_length, slot_len] or None
            )
            
            all_slot_outputs += slot_output
        
        # 计算平均值
        memory_output = all_slot_outputs / self.memory_slots
        
        return memory_output

    def predict_future(self, history_features, current_features, mask=None):
        """
        基于历史特征、当前特征和记忆预测未来特征
        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
        Returns:
            future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
            future_mask: 未来特征的掩码，形状为 [batch_size, future_length]
        """
        batch_size = history_features.shape[0]
        device = history_features.device
        
        # 扩展未来嵌入以匹配批次大小
        future_embed = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接历史特征、当前特征和未来嵌入
        concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

        # 拼接特征的mask
        if mask is not None:
            current_mask = torch.ones(batch_size, current_features.shape[1], 
                                    device=device, dtype=mask.dtype)
            future_mask = torch.ones(batch_size, future_embed.shape[1], 
                                   device=device, dtype=mask.dtype)
            concat_mask = torch.cat([mask, current_mask, future_mask], dim=1)
        else:
            concat_mask = None

        # 从记忆中读取相关信息
        memory_output = self.read_memory(concatenated_features, concat_mask)

        # 用读取了记忆信息的结果做self-attention
        memory_output = self.memory_self_attention(memory_output, memory_output, memory_output)[0]
        
        # 只提取增强特征中的未来部分
        future_part = memory_output[:, -self.future_length:, :]
        
        # 返回预测的未来特征和对应的掩码
        future_mask = torch.ones(batch_size, self.future_length, 
                                device=device, dtype=torch.bool if mask is None 
                                else mask.dtype)
        
        return future_part, future_mask

    def predict_future_direct_crossModal(self, history_features, current_features, RESIDUAL, mask=None):
        """
        基于历史特征、当前特征和记忆预测未来特征
        注意，history——features要用压缩后的特征（确保无mask位）
        基本步骤：
        1. 首先将历史帧和当前帧拼接，然后计算每个slots的相似度
        2. 然后根据相似度选择最相似的slots（topk等）进行fuse（融合）记为fused_memory
        3. 拼接[history_features， current_features， self.future_embedding]（作为Q）与fused_memory（作为KV）做crossAttn得到full_features  
        4. 根据RESIDUAL是否为True，选择是否将full_features与[history_features， current_features]做残差连接（连接的时候只连接历史帧和当前帧对应的部分）
        5. 返回full_features和full_mask

        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            RESIDUAL: 是否使用残差连接
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
        Returns:
            full_features: 预测的未来特征，形状为 [batch_size, history_length+current_length+future_length, d_model]
            full_mask: 未来特征的掩码，形状为 [batch_size, history_length+current_length+future_length]
        """

        # torch.autograd.set_detect_anomaly(True)
        batch_size = history_features.shape[0]
        device = history_features.device
        
        # projected_future = self.future_proj(self.future_embedding)
        # 扩展未来嵌入以匹配批次大小
        # future_embed: [batch_size, future_length, d_model]
        future_embed = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接历史特征、当前特征和未来嵌入
        # concatenated_features: [batch_size, history_length+current_length+future_length, d_model]
        concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

        # 拼接特征的mask
        if mask is not None:
            current_mask = torch.ones(batch_size, current_features.shape[1], 
                                    device=device, dtype=mask.dtype)
            future_mask = torch.ones(batch_size, future_embed.shape[1], 
                                   device=device, dtype=mask.dtype)
            concat_mask = torch.cat([mask, current_mask, future_mask], dim=1)
        else:
            concat_mask = None

        # 拼接历史特征和当前特征，用来计算相似度
        history_current_features = torch.cat([history_features, current_features], dim=1)
        history_current_mask = torch.cat([mask, current_mask], dim=1)

        # 1. 参考update方法中的做法，计算concatenated_features和slots的相似度
        # similarities: [batch_size, memory_slots]
        # similarities = self.compute_sequence_similarity_frobenius(features=concatenated_features, mask=concat_mask)
        # similarities = self.compute_sequence_similarity_pool_cosSim(features=concatenated_features, mask=concat_mask)

        similarities = self.compute_sequence_similarity_frobenius_optimized(features=concatenated_features, mask=concat_mask)
        # similarities = self.compute_sequence_similarity_frobenius_optimized(features=history_current_features, mask=history_current_mask)

        # 2. 选择最相似的slots进行融合
        # topk_similarities: [batch_size, topk], topk_indices: [batch_size, topk]
        topk_similarities, topk_indices = torch.topk(similarities, k=self.topk, dim=1)
        weights = torch.softmax(topk_similarities, dim=1)  # 归一化权重, [batch_size, topk]
        
        topk_slots = torch.stack([self.memory[i] for i in topk_indices], dim=0)  # [batch_size, topk, slot_len, d_model]

        # 创建weights的独立副本，断开梯度依赖（临时流程变动：原本应直接使用weights）
        weights_detached = weights.detach().clone().requires_grad_(True)
        weights_expanded = weights_detached.unsqueeze(-1).unsqueeze(-1)
        fused_memory = (weights_expanded * topk_slots).sum(dim=1)

        # 3. 使用cross attention生成完整特征

        # concatenated_features作为Q，fused_memory作为K和V
        fused_memory_kv = fused_memory  # [batch_size, slot_len, d_model]
        full_features, _ = self.memory_read_attention(
            query=concatenated_features,
            key=fused_memory_kv,
            value=fused_memory_kv,
        )
        
        # 4. 残差连接
        if RESIDUAL:
            # 只对历史和当前特征部分进行残差连接
            history_current_length = history_features.shape[1] + current_features.shape[1]
            full_features = full_features.clone()  # 显式复制，避免in-place操作影响梯度
            full_features[:, :history_current_length] += concatenated_features[:, :history_current_length]

        # return concatenated_features, concat_mask
        return full_features, concat_mask
    
    def predict_future_fcPredict_direct_crossModal(self, history_features, current_features, RESIDUAL, mask=None, res_ori_weight=0.8):
        """
        基于历史特征、当前特征和记忆预测未来特征
        注意，history——features要用压缩后的特征（确保无mask位）
        基本步骤：
        1. 首先将历史帧和当前帧拼接，然后计算每个slots的相似度
        2. 然后根据相似度选择最相似的slots（topk等）进行fuse（融合）记为fused_memory
        3. 拼接[history_features， current_features， self.future_embedding]（作为Q）与fused_memory（作为KV）做crossAttn得到full_features  
            crossAttn昨晚补一个selfAttn
        4. 用一个FC层对full_features做类似预测的操作
        5. 根据RESIDUAL是否为True，选择是否将full_features与[history_features， current_features]做残差连接（连接的时候只连接历史帧和当前帧对应的部分）
        6. 返回full_features和full_mask

        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            RESIDUAL: 是否使用残差连接
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            
        Returns:
            full_features: 预测的未来特征，形状为 [batch_size, history_length+current_length+future_length, d_model]
            full_mask: 未来特征的掩码，形状为 [batch_size, history_length+current_length+future_length]
        """

        # torch.autograd.set_detect_anomaly(True)
        batch_size = history_features.shape[0]
        device = history_features.device

        projected_future = self.future_proj(self.future_embedding)
        # 扩展未来嵌入以匹配批次大小
        # future_embed: [batch_size, future_length, d_model]
        future_embed = projected_future.unsqueeze(0).expand(batch_size, -1, -1)
        # future_embed = self.future_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 拼接历史特征、当前特征和未来嵌入
        # concatenated_features: [batch_size, history_length+current_length+future_length, d_model]
        concatenated_features = torch.cat([history_features, current_features, future_embed], dim=1)

        # 拼接特征的mask
        if mask is not None:
            current_mask = torch.ones(batch_size, current_features.shape[1], 
                                    device=device, dtype=mask.dtype)
            future_mask = torch.ones(batch_size, future_embed.shape[1], 
                                   device=device, dtype=mask.dtype)
            concat_mask = torch.cat([mask, current_mask, future_mask], dim=1)
        else:
            concat_mask = None

        similarities = self.compute_sequence_similarity_frobenius_optimized(features=concatenated_features, mask=concat_mask)

        # 2. 选择最相似的slots进行融合
        topk_similarities, topk_indices = torch.topk(similarities, k=self.topk, dim=1)
        weights = torch.softmax(topk_similarities, dim=1)  # 归一化权重, [batch_size, topk]

        topk_slots = torch.stack([self.memory[i] for i in topk_indices], dim=0)  # [batch_size, topk, slot_len, d_model]

        # 创建weights的独立副本，断开梯度依赖（临时流程变动：原本应直接使用weights）
        weights_detached = weights.detach().clone().requires_grad_(True)
        weights_expanded = weights_detached.unsqueeze(-1).unsqueeze(-1)
        fused_memory = (weights_expanded * topk_slots).sum(dim=1)

        # 3. 使用cross attention生成完整特征
        fused_memory_kv = fused_memory  # [batch_size, slot_len, d_model]
        full_features, _ = self.memory_read_attention(
            query=concatenated_features,
            key=fused_memory_kv,
            value=fused_memory_kv,
        )

        # selfAttn
        full_features, _ = self.memory_self_attention(
            query=full_features,
            key=full_features,
            value=full_features,
        )

        # 4. 用一个FC层，相当于预测头，将ful_features处理一下，形状不变
        full_features = self.fc_prediction_head(full_features)
        
        # 4. 残差连接
        if RESIDUAL:
            # 只对历史和当前特征部分进行残差连接
            history_current_length = history_features.shape[1] + current_features.shape[1]
            full_features = full_features.clone()  # 显式复制
            # res连接的时候需要根据concat_mask的值，有效位置则res连接，无效位置则全用full_features
            # 先初始化一个形状如concat_mask的权重矩阵，然后初始值为res_ori_weight，concat_mask的无效位对应的权重矩阵数值设置为0
            res_weight = torch.zeros_like(concat_mask, dtype=full_features.dtype, device=full_features.device)
            res_weight[concat_mask == 1] = res_ori_weight
            # 扩展res_weight的维度，使其与full_features和concatenated_features的形状一致
            res_weight = res_weight.unsqueeze(-1).expand(-1, -1, full_features.shape[-1])
            full_features[:, :history_current_length] = full_features[:, :history_current_length]*(1-res_weight[:, :history_current_length]) + concatenated_features[:, :history_current_length]*res_weight[:, :history_current_length]

        # 5. 准备一个mask，因为是交互的结果，所以是全有效的
        full_mask = torch.ones(batch_size, full_features.shape[1], 
                                device=device, dtype=concat_mask.dtype)

        return full_features, full_mask
    

    @countTime
    def predict_future_direct_crossModal_andCompress(self, history_features, current_features, mask=None):
        # `H+P+Mf`跨模态交互后，经过压缩（先用Fc层试一下）得到（32*d）的特征作为预测结果
        # [batch_size, history_length + current_length + future_length, d_model]
        raw_res, raw_mask = self.predict_future_direct_crossModal(history_features=history_features, current_features=current_features, mask=mask, RESIDUAL=False)
        # [batch_size, d_model, history_length + current_length + future_length]
        transposed_res = raw_res.transpose(1, 2)
        # [batch_size, d_model, future_length]
        compressed_res = self.fc_compressor(transposed_res)
        # [batch_size, future_length, d_model]
        predicted_res = compressed_res.transpose(1, 2)

        # raw_mask的后future_length取出来作为预测结果的mask
        predicted_mask = raw_mask[:, -self.future_length:]

        return predicted_res, predicted_mask

    @countTime
    def forward(self, history_features, current_features, mask=None, short_start=None, vid=None):
        """
        前向传播：更新记忆槽并预测未来特征
        每经过optimization_interval次调用后，自动执行记忆多样性优化
        
        Args:
            history_features: 历史特征，形状为 [batch_size, history_length, d_model]
            current_features: 当前特征，形状为 [batch_size, current_length, d_model]
            mask: 历史特征的有效性掩码，形状为 [batch_size, history_length]
            short_start: 
            vid: 保留参数
            
        Returns:
            future_features: 预测的未来特征，形状为 [batch_size, future_length, d_model]
            future_mask: 未来特征的掩码，形状为 [batch_size, future_length]
        """
        # 该版本保持 legacy 行为：默认返回 full 序列（H+C+F）
        return self.forward_with_controls(
            history_features=history_features,
            current_features=current_features,
            mask=mask,
            short_start=short_start,
            vid=vid,
        )

    def update_memory_with_delta_gt(
        self,
        history_features,
        delta_gt,
        delta_gt_valid=None,
        history_mask=None,
        selected_slots=None,
        momentum=0.1,
    ):
        """使用 delta_gt 对检索到的槽进行监督更新（Step4）。

        输入:
            history_features: 历史特征，[B, H, D]，用于检索最相关槽位。
            delta_gt: 真实位移监督，[B, T, D]。
            delta_gt_valid: 位移有效位，[B, T]，可选。
            history_mask: 历史有效位，[B, H]，可选。
            selected_slots: 预先选定的槽索引，[B]，可选。
            momentum: EMA 更新比例，范围 (0, 1]。
        输出:
            None。函数会原地更新 memory 的前 T 段。
        """
        try:
            if delta_gt is None:
                return
            with torch.no_grad():
                batch_size, delta_len, feat_dim = delta_gt.shape
                if feat_dim != self.d_model:
                    raise ValueError("delta_gt 的特征维度与 d_model 不一致。")
                if history_mask is None:
                    history_mask = torch.ones(
                        history_features.shape[0],
                        history_features.shape[1],
                        device=history_features.device,
                        dtype=torch.bool,
                    )
                if selected_slots is None:
                    similarity = self.compute_sequence_similarity_frobenius_optimized(
                        history_features,
                        history_mask,
                    )
                    selected_slots = torch.argmax(similarity, dim=1)
                update_len = min(self.slot_len, delta_len)
                for b in range(batch_size):
                    slot_idx = selected_slots[b]
                    valid_mask = torch.ones(
                        update_len,
                        device=delta_gt.device,
                        dtype=torch.bool,
                    )
                    if delta_gt_valid is not None:
                        valid_mask = delta_gt_valid[b, :update_len].bool()
                    if not valid_mask.any():
                        continue
                    current_slot = self.memory.data[slot_idx, :update_len]
                    delta_slice = delta_gt[b, :update_len]
                    alpha = valid_mask.float().unsqueeze(-1) * momentum
                    updated_slot = current_slot * (1 - alpha) + delta_slice * alpha
                    self.memory.data[slot_idx, :update_len] = updated_slot
                    self.slot_usage_counter[slot_idx] += 1
        except Exception as err:
            print(f"update_memory_with_delta_gt error: {err}")
            raise

    def select_slots_by_text_query(self, text_query, text_query_mask=None):
        """基于文本语义选择最相关槽位。

        输入:
            text_query: 文本特征，[B, Lq, D] 或 [B, D]。
            text_query_mask: 文本有效位，[B, Lq]，可选。
        输出:
            selected_slots: 每个样本的槽索引，[B]。
        """
        if text_query is None:
            return None

        use_slot_mean = (
            self.legacy_v1 or self.slot_select_mode == "slot_mean_text"
        )
        if use_slot_mean:
            query_seq = text_query if text_query.dim() == 3 else text_query.unsqueeze(1)
            if text_query_mask is None:
                text_query_mask = torch.ones(
                    query_seq.shape[0],
                    query_seq.shape[1],
                    device=query_seq.device,
                    dtype=torch.bool,
                )
            slot_similarity = self.compute_sequence_similarity_pool_cosSim(
                features=query_seq,
                mask=text_query_mask,
            )
            return torch.argmax(slot_similarity, dim=1)

        if self.slot_select_mode != "header":
            raise ValueError("slot_select_mode 必须是 'header' 或 'slot_mean_text'。")

        if text_query.dim() == 3:
            query_repr = text_query.mean(dim=1)
        else:
            query_repr = text_query
        normalized_query = F.normalize(query_repr, p=2, dim=-1)
        normalized_header = F.normalize(self.header_tokens, p=2, dim=-1)
        semantic_scores = torch.matmul(normalized_query, normalized_header.transpose(0, 1))
        return torch.argmax(semantic_scores, dim=1)

    # ================================================================
    #  Step 2: Local Phase Alignment — 滑窗相位对齐
    # ================================================================

    def compute_phase_index(self, m_intra_delta, selected_slots):
        """在选定 slot 上用宽度 T_c 的滑窗做余弦相关，找到最匹配起点 k*。

        输入:
            m_intra_delta: 压缩器动态通道 [B, T_c, D]。
            selected_slots: Step1 选出的槽索引 [B]。
        输出:
            k_star: 每个样本在对应 slot 中的最佳匹配窗口起点 [B]。
        """
        B = m_intra_delta.shape[0]
        T_c = m_intra_delta.shape[1]
        n = self.future_length
        device = m_intra_delta.device

        slots = self.memory[selected_slots]                       # [B, K, D]
        K = slots.shape[1]
        max_start = K - T_c - n                                   # k*+T_c+n <= K

        if max_start <= 0:
            return torch.zeros(B, dtype=torch.long, device=device)

        # unfold: [B, K-T_c+1, D, T_c] -> permute -> [B, num_win, T_c, D]
        windows = slots.unfold(1, T_c, 1).permute(0, 1, 3, 2)
        windows = windows[:, :max_start + 1]                      # [B, W, T_c, D]

        query = m_intra_delta.unsqueeze(1)                        # [B, 1, T_c, D]
        corr = F.cosine_similarity(query, windows, dim=-1)        # [B, W, T_c]
        corr_avg = corr.mean(dim=-1)                              # [B, W]
        k_star = torch.argmax(corr_avg, dim=1)                    # [B]
        return k_star

    def extract_delta_read(self, selected_slots, k_star):
        """从选定 slot 的 k*+T_c 位置提取长度为 n 的 Δ_read 段。

        输入:
            selected_slots: 槽索引 [B]。
            k_star: 滑窗匹配起点 [B]。
        输出:
            delta_read: [B, n, D]，slot 中匹配窗口右侧的位移趋势。
        """
        B = selected_slots.shape[0]
        T_c = self.compress_len
        n = self.future_length
        device = self.memory.device

        slots = self.memory[selected_slots]                       # [B, K, D]
        K = slots.shape[1]

        delta_list = []
        for b in range(B):
            start = k_star[b].item() + T_c
            end = min(start + n, K)
            seg = slots[b, start:end]                             # [<=n, D]
            if seg.shape[0] < n:
                pad = torch.zeros(n - seg.shape[0], self.d_model, device=device)
                seg = torch.cat([seg, pad], dim=0)
            delta_list.append(seg)
        return torch.stack(delta_list, dim=0)                     # [B, n, D]

    # ================================================================
    #  Step 3: Conditional Propagation — 条件化未来外推
    # ================================================================

    def predict_future_4step(self, m_intra_static, delta_read, current_features=None):
        """基于三公式实现条件化未来特征外推。

        公式1: F_ref = [M_intra_static ‖ F_c ‖ F_p] 
        公式2: H = CrossAttn(Q=F_ref, K=Δ_read, V=Δ_read)
        公式3: F_f = F_p + MLP([H_tail ‖ F_ref_tail])  (特征拼接)

        输入:
            m_intra_static: 压缩器静态通道 [B, T_c, D]。
            delta_read: slot 中提取的位移趋势 [B, n, D]。
            current_features: 当前时窗特征 [B, L_c, D]，可选。
        输出:
            F_f: 预测的未来特征 [B, n, D]。
        """
        B = m_intra_static.shape[0]
        n = delta_read.shape[1]

        F_p = self.future_proj(self.future_embedding)             # [n, D]
        F_p = F_p.unsqueeze(0).expand(B, -1, -1)                 # [B, n, D]

        # 在 4-step 中拼接当前段特征，增强与 Δ_read 的交互上下文。
        ref_parts = [m_intra_static]
        if current_features is not None:
            ref_parts.append(current_features)
        ref_parts.append(F_p)
        F_ref = torch.cat(ref_parts, dim=1)                      # [B, T_c(+L_c)+n, D]

        H, _ = self.step3_cross_attn(
            query=F_ref, key=delta_read, value=delta_read,
        )                                                         # [B, T_c(+L_c)+n, D]

        H_tail = H[:, -n:]                                       # [B, n, D]
        F_ref_tail = F_ref[:, -n:]                                # [B, n, D]
        fused = torch.cat([H_tail, F_ref_tail], dim=-1)           # [B, n, 2D]
        F_f = F_p + self.step3_mlp(fused)                         # [B, n, D]
        return F_f

    # ================================================================
    #  Step 4: Structural Memory Update — 结构保持更新
    # ================================================================

    def update_memory_step4(self, selected_slots, k_star, text_query,
                            delta_gt, delta_gt_valid):
        """训练期按 k* 驱动 header EMA + slider EMA 更新。

        公式:
            header:  h_j* = (1-λ) h_j* + λ F_q
            slider:  S_j*[k*:k*+n] = (1-λ) S + λ Δ_gt

        输入:
            selected_slots: 槽索引 [B]。
            k_star: 匹配起点 [B]。
            text_query: 文本特征 [B, Lq, D] 或 [B, D]。
            delta_gt: 真实位移 [B, T, D]。
            delta_gt_valid: 有效位 [B, T]。
        """
        if not self.training:
            return

        lam = self.ema_lambda
        B = selected_slots.shape[0]

        with torch.no_grad():
            # ── header EMA ──
            if text_query is not None:
                F_q = text_query.mean(dim=1) if text_query.dim() == 3 else text_query
                for b in range(B):
                    j = selected_slots[b]
                    self.header_tokens.data[j] = (
                        (1 - lam) * self.header_tokens.data[j]
                        + lam * F_q[b].detach()
                    )

            # ── slider EMA: S[k*:k*+n] ──
            if delta_gt is not None:
                n = self.future_length
                for b in range(B):
                    j = selected_slots[b]
                    k = k_star[b].item()
                    update_len = min(n, delta_gt.shape[1], self.slot_len - k)
                    if update_len <= 0:
                        continue
                    if delta_gt_valid is not None:
                        vm = delta_gt_valid[b, :update_len].float().unsqueeze(-1)
                    else:
                        vm = torch.ones(update_len, 1, device=delta_gt.device)
                    self.memory.data[j, k:k + update_len] = (
                        (1 - lam * vm) * self.memory.data[j, k:k + update_len]
                        + lam * vm * delta_gt[b, :update_len].detach()
                    )
                    self.slot_usage_counter[j] += 1

    def compute_memory_constraint_loss(self):
        """计算记忆约束损失 L_mem = L_orth(h) + L_consist(S)，loss_struct单独计算。

        L_orth:  header 间正交性——抑制不同 slot header 坍缩。
        L_consist: slot 内时序平滑性——防止动作流形断裂。

        输出:
            loss: 标量张量。
        """
        M = self.memory_slots

        # L_orth: off-diagonal cosine similarity penalty
        H_norm = F.normalize(self.header_tokens, p=2, dim=-1)     # [M, D]
        sim = torch.matmul(H_norm, H_norm.transpose(0, 1))       # [M, M]
        eye = torch.eye(M, device=sim.device)
        L_orth = ((sim - eye) ** 2).sum() / max(M * (M - 1), 1)

        # L_consist: adjacent-frame difference penalty
        diffs = self.memory[:, 1:] - self.memory[:, :-1]         # [M, K-1, D]
        L_consist = (diffs ** 2).mean()

        return L_orth + L_consist

    def compute_step4_structural_loss(
        self,
        selected_slots,
        delta_gt,
        delta_gt_valid=None,
    ):
        """计算 Step4 结构约束损失（槽内位移与真实位移对齐）。

        输入:
            selected_slots: 每个样本匹配的槽索引，[B]。
            delta_gt: 真实位移监督，[B, T, D]。
            delta_gt_valid: 位移有效位，[B, T]，可选。
        输出:
            loss_struct: 标量张量，越小表示记忆位移与监督越一致。
        """
        if selected_slots is None or delta_gt is None:
            return self.memory.sum() * 0.0
        update_len = min(self.slot_len, delta_gt.shape[1])
        selected_memory = self.memory[selected_slots, :update_len, :]
        target_delta = delta_gt[:, :update_len, :]
        if delta_gt_valid is None:
            valid_mask = torch.ones(
                target_delta.shape[0],
                update_len,
                device=target_delta.device,
                dtype=target_delta.dtype,
            )
        else:
            valid_mask = delta_gt_valid[:, :update_len].float().to(target_delta.device)
        loss_raw = F.smooth_l1_loss(
            selected_memory,
            target_delta,
            reduction="none",
        )
        weighted_loss = loss_raw * valid_mask.unsqueeze(-1)
        denom = valid_mask.sum() * target_delta.shape[-1]
        denom = torch.clamp(denom, min=1.0)
        return weighted_loss.sum() / denom

    def forward_with_controls(
        self,
        history_features,
        current_features,
        mask=None,
        short_start=None,
        vid=None,
        history_static=None,
        history_delta=None,
        text_query=None,
        text_query_mask=None,
        delta_gt=None,
        delta_gt_valid=None,
        output_mode="legacy_full",
        allow_infer_memory_update=False,
    ):
        """可控前向：自动选择 4-step 路径或 legacy 路径。

        当 history_static 和 history_delta 同时存在（compressor 为 dual_intra 模式）
        时自动走完整 4-step 路径；否则回退到 legacy 路径。

        输入:
            history_features: 压缩后历史特征 [B, T_c, D]。
            current_features: 当前短时窗特征 [B, L_c, D]。
            mask: history_features 有效位 [B, T_c]，可选。
            history_static: compressor 静态通道 [B, T_c, D]，4-step 用。
            history_delta: compressor 动态通道 [B, T_c, D]，4-step 用。
            text_query: 文本特征 [B, Lq, D] 或 [B, D]，Step1 用。
            text_query_mask: 文本有效位 [B, Lq]，slot_mean_text 模式可选。
            delta_gt: 训练期真实位移 [B, T, D]，Step4 用。
            delta_gt_valid: delta_gt 有效位 [B, T]，可选。
            output_mode: 'future_only' 或 'legacy_full'。
            allow_infer_memory_update: 推理期是否更新 memory。
        输出:
            (features, mask): 由 output_mode 决定返回序列。
        """
        B = history_features.shape[0]
        device = history_features.device
        use_4step = (
            not self.legacy_v1
            and history_delta is not None
            and history_static is not None
        )

        # ── Step1: 文本驱动槽选择（两条路径共用） ──
        selected_slots = self.select_slots_by_text_query(
            text_query=text_query,
            text_query_mask=text_query_mask,
        )
        if selected_slots is None:
            history_for_sim = history_features
            if history_static is not None:
                history_for_sim = history_static
            slot_similarity = self.compute_sequence_similarity_frobenius_optimized(
                history_for_sim, mask,
            )
            selected_slots = torch.argmax(slot_similarity, dim=1)

        if use_4step:
            # ============================================================
            #  4-step 路径
            # ============================================================

            # Step2: 滑窗相位对齐 → k*（可被消融为固定起点）
            if self.disable_phase_alignment:
                max_start = self.slot_len - self.compress_len - self.future_length
                bounded_start = max(0, min(self.fixed_phase_start, max(0, max_start)))
                k_star = torch.full(
                    (B,),
                    bounded_start,
                    dtype=torch.long,
                    device=device,
                )
            else:
                k_star = self.compute_phase_index(history_delta, selected_slots)

            # Step3: 提取 Δ_read，三公式条件外推 → F_f
            delta_read = self.extract_delta_read(selected_slots, k_star)
            F_f = self.predict_future_4step(
                m_intra_static=history_static,
                delta_read=delta_read,
                current_features=current_features,
            )

            # Step4: 训练期 EMA 更新 header + slider
            if self.training and delta_gt is not None:
                self.update_memory_step4(
                    selected_slots, k_star, text_query,
                    delta_gt, delta_gt_valid,
                )

            # 默认监督来源改为真实压缩动态结果 history_delta（仅 4-step 路径可用）。
            # 维度对齐策略：
            # 1) 时间维：按 compute_step4_structural_loss 内部 min(...) 自动截断；
            # 2) 特征维：仅在 D 与 memory 一致时启用该监督，否则退回纯约束损失。
            target_delta = None
            target_delta_valid = None
            if history_delta is not None and history_delta.shape[-1] == self.d_model:
                target_delta = history_delta.detach()
                if mask is not None:
                    target_delta_valid = mask
            elif delta_gt is not None and delta_gt.shape[-1] == self.d_model:
                target_delta = delta_gt.detach()
                target_delta_valid = delta_gt_valid

            struct_align_loss = self.memory.sum() * 0.0
            if target_delta is not None:
                struct_align_loss = self.compute_step4_structural_loss(
                    selected_slots=selected_slots,
                    delta_gt=target_delta,
                    delta_gt_valid=target_delta_valid,
                )

            # 保留原有约束项，避免影响已有 memory 正则化行为。
            self.last_memory_struct_loss = (
                self.compute_memory_constraint_loss() + struct_align_loss
            )

            future_mask = torch.ones(B, F_f.shape[1], device=device)
            if output_mode == "future_only":
                return F_f, future_mask

            # legacy_full: 拼接 history + current + future
            cur_mask = torch.ones(B, current_features.shape[1], device=device)
            hist_mask = mask if mask is not None else torch.ones(
                B, history_features.shape[1], device=device,
            )
            full_features = torch.cat(
                [history_features, current_features, F_f], dim=1,
            )
            full_mask = torch.cat([hist_mask, cur_mask, future_mask], dim=1)
            return full_features, full_mask

        else:
            # ============================================================
            #  Legacy 路径（无结构化 static/delta 输入时回退）
            # ============================================================
            history_for_memory = history_features
            if history_static is not None:
                history_for_memory = history_static
            elif history_delta is not None:
                history_for_memory = history_delta

            step4_struct_loss = self.compute_step4_structural_loss(
                selected_slots=selected_slots,
                delta_gt=delta_gt,
                delta_gt_valid=delta_gt_valid,
            )
            self.last_memory_struct_loss = step4_struct_loss

            should_update = self.training or allow_infer_memory_update
            if should_update:
                self.update_memory_normal_topk(
                    features=history_for_memory, mask=mask,
                )
            if self.training and delta_gt is not None:
                self.update_memory_with_delta_gt(
                    history_features=history_for_memory,
                    delta_gt=delta_gt,
                    delta_gt_valid=delta_gt_valid,
                    history_mask=mask,
                    selected_slots=selected_slots,
                )

            full_features, full_mask = self.predict_future_fcPredict_direct_crossModal(
                history_features=history_for_memory,
                mask=mask,
                current_features=current_features,
                RESIDUAL=True,
                res_ori_weight=0.7,
            )
            if output_mode == "future_only":
                return (
                    full_features[:, -self.future_length:],
                    full_mask[:, -self.future_length:],
                )
            if output_mode == "legacy_full":
                return full_features, full_mask
            raise ValueError(
                "output_mode 必须是 'legacy_full' 或 'future_only'。"
            )

    def set_auto_optimization(self, diversity_loss_weight=None, 
                            optimization_interval=None, optimizer_lr=None):
        """
        动态设置自动优化参数
        
        Args:
            diversity_loss_weight: 多样性损失权重
            optimization_interval: 优化间隔
            optimizer_lr: 优化器学习率
        """
        
        if diversity_loss_weight is not None:
            self.diversity_loss_weight = diversity_loss_weight
            
        if optimization_interval is not None:
            self.optimization_interval = optimization_interval
            
        if optimizer_lr is not None:
            # 重新创建优化器
            self.memory_optimizer = torch.optim.Adam([self.memory], lr=optimizer_lr)
            
        print(f"Auto optimization settings updated: "
              f"loss_weight={self.diversity_loss_weight}, "
              f"interval={self.optimization_interval}")

    # 使用示例和说明
    """
    使用方法：

    1. 基本初始化：
    memory_module = InterMemorySeq(
        d_model=256,
        future_length=32,
        memory_slots=64,
        slot_len=64,
        diversity_loss_weight=0.5,    # 多样性损失权重
        optimization_interval=10,     # 每10次forward后优化一次
    )

    2. 在训练循环中自动优化：
    for batch in dataloader:
        history_features, current_features, mask = batch
        
        # forward会自动更新记忆并进行周期性优化
        future_pred, future_mask = memory_module(history_features, current_features, mask)
        
        # 使用预测结果计算主任务损失
        main_loss = criterion(future_pred, future_target)
        main_loss.backward()
        optimizer.step()

    3. 动态调整优化参数：
    memory_module.set_auto_optimization(
        diversity_loss_weight=0.2,   # 降低多样性损失权重
        optimization_interval=20,     # 增加优化间隔
        optimizer_lr=5e-5            # 调整优化器学习率
    )
    """


