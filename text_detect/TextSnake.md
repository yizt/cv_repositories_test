


工程地址：[TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch)

```bash
mkdir save/pretrained
cd save/pretrained
ln -s /sdb/tmp/pretrained_model/textsnake_vgg_180.pth textsnake_vgg_180.pth
```

## 预测
a) 修改`demo.py`文件
  
  在文件头增加`import multiprocessing as mp`,在`main`函数第一行增加`mp.set_start_method('spawn')`

b）执行如下命令预测
```bash
EXPNAME=pretrained
CUDA_VISIBLE_DEVICES=0 
python demo.py $EXPNAME --checkepoch 180 --img_root /sdb/tmp/tower_section/test_dataset/images
```


## 训练
a) 准备数据,`data/total-text`目录结构如下
```bash
Images
  --Train
    --img1001.jpg
    ...
  --Test
    --img1.jpg
    ...
gt
  --Train
    --poly_gt_img1001.mat
    ...
  --Test
    --poly_gt_img1.mat
    ...
```

b) 修改`train_textsnake.py+`，使用命令`git diff train_textsnake.py` 修改如下：
```python
 from util.shedule import FixLR
+import multiprocessing as mp

 def save_model(model, epoch, lr, optimzer):
+    model = model.module if cfg.mgpu else model

-        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
+        'model': model.state_dict(),

-    model.load_state_dict(state_dict['model'])
+    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict['model'].items()}
+    model.load_state_dict(new_state_dict)

 def main():
+    mp.set_start_method('spawn')

     # Model
     model = TextNet(is_training=True, backbone=cfg.net)
+    if cfg.resume:
+        load_model(model, cfg.resume)

-    if cfg.resume:
-        load_model(model, cfg.resume)
-
     criterion = TextLoss()
```

c) 训练
```bash
EXPNAME=tt
CUDA_VISIBLE_DEVICES=0,1,2,3
python train_textsnake.py $EXPNAME --viz --batch_size 8 --mgpu \
--resume /sdb/tmp/pretrained_model/textsnake_vgg_0.pth
```


##总结
1. 总体上工程代码不成熟，多处报错，需要修改才能正确运行