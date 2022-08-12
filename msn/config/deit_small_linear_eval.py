custom_imports = dict(
    imports=[
        'model.msn_vit',
        'model.benchmark.linear_eval.msn_cls_head',
    ],
    allow_failed_imports=False
)

# dataset settings
data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],  # Mean values used to pre-training the pre-trained backbone models
    std=[0.229, 0.224, 0.225])  # Standard variance used to pre-training the pre-trained backbone models


train_pipeline=[
    dict(type='RandomResizedCrop',size=224,scale=(0.08,1.0)),
    dict(type='RandomHorizontalFlip'),
]

val_pipeline = [
    dict(type='Resize',size=256),
    dict(type='CenterCrop',size=224),
    
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    val_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])


# dataset summary
data = dict(
    imgs_per_gpu=512,  # total 32x8=256, 8GPU linear cls
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/home/intern/dataset/imagenet/train',
            ann_file='/home/intern/scratch/likaixin/mmmsn/work_dir/imagenet_meta/train.txt',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/home/intern/dataset/imagenet/val',
            ann_file='/home/intern/scratch/likaixin/mmmsn/work_dir/imagenet_meta/val_mmlab.txt',
        ),
        pipeline=val_pipeline,
        prefetch=prefetch))


model = dict(
    type='Classification',
    backbone=dict(type='MSNVisionTransformer',  # deit_small
                  embed_dim=384,
                  depth=12,
                  num_heads=6,
                  patch_size=16,
                  mlp_ratio=4.,
                  frozen_stages=12,
                  qkv_bias=True,
                  no_final_norm=True,
                  ),
    head=dict(
        type='MSNClsHead',
        in_channels=384,
        num_classes=1000,
        vit_backbone=False, # mmlab的实现中返回所有token。msn只返回了cls token，所以置为false
    ))


# optimizer
optimizer = dict(type='SGD', nesterov=True, lr=6.4, momentum=0.9, weight_decay=0.0)
optimizer_config=dict()

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0., by_epoch=False)

evaluation = dict(interval=5, topk= (1, 5, ))  # For ImageNet Dataset evaluation.

checkpoint_config = dict(interval=1, max_keep_ckpts=5)
# runtime settings

runner = dict(type='EpochBasedRunner', max_epochs=100)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='MSN Linear Eval',
                #id='240ttg0n',
                #resume='allow',
                config=dict(
                    model=model,
                    train_pipeline=train_pipeline,
                    test_pipeline=val_pipeline,
                    epochs=runner['max_epochs'],
                    batch_size=data['imgs_per_gpu'],
                    optimizer=optimizer,
                    optimizer_config=optimizer_config,
                    lr_config=lr_config
                    ),
                )
            )
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None  # 这部分是runner的配置，作用于整个task，与模型的预训练加载无关
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
find_unused_parameters = True # necessary when freezing part of the model