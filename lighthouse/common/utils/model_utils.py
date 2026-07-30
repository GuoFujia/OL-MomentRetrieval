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
import torch
import copy
from typing import Iterable, List, Tuple


# InterMemory 模块参数前缀（鲁棒性 fine-tune 仅更新此模块）
DEFAULT_INTER_MEMORY_PREFIXES: Tuple[str, ...] = ("inter_memory.",)


def is_inter_memory_param(param_name: str, prefixes: Iterable[str] = DEFAULT_INTER_MEMORY_PREFIXES) -> bool:
    """判断参数是否属于 InterMemory 相关模块。

    输入:
        param_name: 模型参数全名（model.named_parameters 的 name）
        prefixes: 可训练模块前缀元组
    输出:
        bool, True 表示该参数应参与 InterMemory-only 训练
    """
    return any(param_name.startswith(prefix) for prefix in prefixes)


def apply_train_inter_memory_only(
    model,
    prefixes: Iterable[str] = DEFAULT_INTER_MEMORY_PREFIXES,
    logger=None,
) -> Tuple[int, int]:
    """冻结除 InterMemory 相关模块外的全部参数。

    输入:
        model: PyTorch 模型
        prefixes: 保持可训练的参数名前缀
        logger: 可选日志对象
    输出:
        (frozen_count, trainable_count): 冻结与可训练参数量
    """
    prefix_tuple = tuple(prefixes)
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if is_inter_memory_param(name, prefix_tuple):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    message = (
        f"train_inter_memory_only: frozen {frozen_count:,} params, "
        f"trainable {trainable_count:,} params"
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)

    return frozen_count, trainable_count


def get_trainable_param_groups(model) -> List[dict]:
    """构建仅含 requires_grad=True 参数的 optimizer param group。

    输入:
        model: PyTorch 模型
    输出:
        param_dicts: AdamW 等优化器可用的 param group 列表
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return [{"params": trainable_params}]


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