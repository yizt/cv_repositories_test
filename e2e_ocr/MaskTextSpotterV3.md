[TOC]



工程地址：<https://github.com/MhLiao/MaskTextSpotterV3>



## 预测

### IC15



```shell
mkdir datasets
cd datasets
ln -s /home/mydir/dataset/IC15 icdar2015

cd /home/mydir/dataset/IC15
ln -s ch4_test_images test_images
```





```shell
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"
python tools/test_net.py --config-file configs/mixtrain/seg_rec_poly_fuse_feature.yaml \
--local_rank 1 \
MODEL.WEIGHT /home/mydir/pretrained_model/trained_model.pth \
INPUT.MIN_SIZE_TEST 1000 \
DATASETS.TEST "('icdar_2015_test',)"

python tools/test_net.py --config-file configs/mixtrain/seg_rec_poly_fuse_feature.yaml \
--local_rank 1 \
MODEL.WEIGHT /home/mydir/pretrained_model/trained_model.pth \
INPUT.MIN_SIZE_TEST 1440 \
DATASETS.TEST "('icdar_2015_test',)"
```



## 评估



```shell
cd evaluation
ln -s /home/mydir/dataset/evaluation/lexicons lexicons

apt-get install zip
```



### IC15

```shell
cd evaluation/icdar2015/e2e
python script.py
```

结果:

```shell
score_det: score_det: 0.01 score_rec: 0.4 score_rec_seq: 0.7 lexicon_type: 3 weighted_ed: True use_seq: True use_char: True mix: True
Calculated!{"precision": 0.9221014492753623, "recall": 0.7351949927780452, "hmean": 0.8181087597106885, "AP": 0}


score_det: score_det: 0.01 score_rec: 0.4 score_rec_seq: 0.7 lexicon_type: 2 weighted_ed: True use_seq: True use_char: True mix: True
Calculated!{"precision": 0.8440046565774156, "recall": 0.6981222917669716, "hmean": 0.764163372859025, "AP": 0}

score_det: score_det: 0.01 score_rec: 0.4 score_rec_seq: 0.7 lexicon_type: 1 weighted_ed: True use_seq: True use_char: True mix: True
Calculated!{"precision": 0.7680995475113123, "recall": 0.6538276360134809, "hmean": 0.706371911573472, "AP": 0}
```

论文中使用1440(GPU显存不够)，HMean分别为83.1、79.1、75.1



## 异常记录

1) RuntimeError: unable to write to file </torch_1525_3932658534>

在tools/test_net.py最开始增加如下代码

```python
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]
```

 

参考：	<https://bbs.huaweicloud.com/forum/thread-17989-1-1.html>

2) RuntimeError: CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate 226.00 MiB (GPU 0; 7.93 GiB total capacity; 4.21 GiB already allocated; 102.94 MiB free; 4.65 GiB reserved in total by PyTorch)
```

INPUT.MIN_SIZE_TEST 改小

3)