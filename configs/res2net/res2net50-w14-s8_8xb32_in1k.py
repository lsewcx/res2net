_base_ = [
    # '../_base_/models/res2net50-w14-s8.py',
    # '../_base_/datasets/imagenet_bs32_pil_resize.py',
    # '../_base_/schedules/imagenet_bs256.py',
    "../_base_/default_runtime.py"
]

model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="Res2Net",
        depth=50,
        scales=8,
        base_width=14,
        deep_stem=False,
        avg_down=False,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=5,
        in_channels=2048,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1, 5),
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth",
    ),
)

# dataset settings
dataset_type = "ImageNet"
data_preprocessor = dict(
    num_classes=5
    ,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", scale=256, backend="pillow"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=256, edge="short", backend="pillow"),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root="/kaggle/working/res2net/Imagenet2",
        split="train",
        classes=[
            "N47",
            "S12",
            "S36",
            "S19",
            "YDN6",
        ],
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root="/kaggle/working/res2net/Imagenet2",
        split="val",
        classes=[
            "N47",
            "S12",
            "S36",
            "S19",
            "YDN6",
        ],
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="Accuracy", topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    # 使用 SGD 优化器来优化参数
    optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
)

# 学习率参数的调整策略
# 'MultiStepLR' 表示使用多步策略来调度学习率（LR）。
param_scheduler = dict(
    type="MultiStepLR", by_epoch=True, milestones=[30, 60, 90], gamma=0.1
)

# 训练的配置，迭代 100 个 epoch，每一个训练 epoch 后都做验证集评估
# 'by_epoch=True' 默认使用 `EpochBaseLoop`,  'by_epoch=False' 默认使用 `IterBaseLoop`
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
# 使用默认的验证循环控制器
val_cfg = dict()
# 使用默认的测试循环控制器
test_cfg = dict()

# 通过默认策略自动缩放学习率，此策略适用于总批次大小 256
# 如果你使用不同的总批量大小，比如 512 并启用自动学习率缩放
# 我们将学习率扩大到 2 倍
auto_scale_lr = dict(base_batch_size=256)


# defaults to use registries in mmpretrain
default_scope = "mmpretrain"

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type="IterTimerHook"),
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type="ParamSchedulerHook"),
    # save checkpoint per epoch.
    checkpoint=dict(type="CheckpointHook", interval=1),
    # checkpoint=dict(type='CheckpointHook', interval=1, save_best="auto", max_keep_ckpt=3),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # validation results visualization, set True to enable it.
    visualization=dict(type="VisualizationHook", enable=False),
)


# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
# vis_backends = [dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]
vis_backends = [dict(type="LocalVisBackend")]

visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)

# set log level
log_level = "INFO"

# load from which checkpoint
# load_from ='E:/configs/_base_/pre_train/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth'

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
