_base_ = [
    'base_imports.py',
]
# meta
num_rand_views = 1
num_focal_views = 4
rand_size = 224
rand_crop_scale = (0.3, 1.0)
focal_size = 96
focal_crop_scale = (0.05, 0.3)
color_jitter_strength = 0.5

# data settings
data_source = 'ImageNet'
dataset_type = 'MultiViewDataset'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],  # Mean values used to pre-training the pre-trained backbone models
    std=[0.229, 0.224, 0.225])  # Standard variance used to pre-training the pre-trained backbone models

# pipelines
color_distort_transform = [
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(type='ColorJitter',
                 brightness=0.8 * color_jitter_strength,
                 contrast=0.8 * color_jitter_strength,
                 saturation=0.8 * color_jitter_strength,
                 hue=0.2 * color_jitter_strength, )
        ], p=0.8),
    dict(type='RandomGrayscale', p=0.2),
]
rand_view_pipeline = [
    dict(type='RandomResizedCrop', size=rand_size, scale=rand_crop_scale),
    dict(type='RandomHorizontalFlip'),
    *color_distort_transform,
    dict(type='GaussianBlur', p=0.5, sigma_min=0.1, sigma_max=2.0),
]
focal_view_pipeline = [
    dict(type='RandomResizedCrop', size=focal_size, scale=focal_crop_scale),
    dict(type='RandomHorizontalFlip'),
    *color_distort_transform,
    dict(type='GaussianBlur', p=0.5, sigma_min=0.1, sigma_max=2.0),
]

# prefetch
prefetch = False
prefetch_suffix = [dict(type='ToTensor'),
                   dict(type='Normalize', **img_norm_cfg)]
if not prefetch:
    rand_view_pipeline.extend(prefetch_suffix)
    focal_view_pipeline.extend(prefetch_suffix)

# dataset summary
data = dict(
    samples_per_gpu=2048,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/home/intern/dataset/imagenet/train',
            ann_file='/home/intern/scratch/likaixin/mmmsn/work_dir/imagenet_meta/train.txt',
        ),
        num_views=[num_rand_views + 1, num_focal_views],  # MultiViewDataset会自动按顺序对每个pipeline重复指定次
        pipelines=[rand_view_pipeline, focal_view_pipeline],
        prefetch=prefetch,
    ))

# model
embed_dim = 128
hidden_dim = 512  # Used in model neck
output_dim = 128

model = dict(
    type='MSN',

    output_dim=output_dim,
    patch_drop=0.15,
    num_proto=512,
    start_sharpen=0.25,
    final_sharpen=0.25,
    freeze_proto=False,

    backbone=dict(type='MSNVisionTransformer',
                  embed_dim=embed_dim,
                  depth=6,
                  num_heads=2,
                  patch_size=32,
                  mlp_ratio=4.,
                  ),
    neck=dict(
        type='MSNNeck',
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        emb_dim=embed_dim,
        use_bn=True,
    ),
    head=dict(
        type='MSNHead',
        tau=0.1,
        num_views=num_focal_views + num_rand_views,
        me_max=True,
        me_max_weight=1,
        use_entropy=True,
        ent_weight=1,
        use_sinkhorn=True,
    ),
)

# Schedule


# optimizer
# TODO: param group
"""
param_groups = [
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'WD_exclude': True,
         'weight_decay': 0}
    ]
    if prototypes is not None:
        param_groups.append({
            'params': [prototypes],
            'lr': ref_lr,
            'LARS_exclude': True,
            'WD_exclude': True,
            'weight_decay': 0
        })
"""

# learning policy
start_lr = 0.0002
lr = 0.001
final_lr = 1.0e-06
final_weight_decay = 0.4
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=True,
    min_lr_ratio=final_lr / lr,
    warmup='linear',
    warmup_ratio=start_lr / lr,
    warmup_iters=15,  # When by_epoch is set, this means the number of warmup epochs
    warmup_by_epoch=False)

optimizer = dict(type='AdamW', lr=lr, paramwise_options={
    '.*(bias|bn).*': dict(WD_exclude=True, weight_decay=0),  # len(p.shape)=1怎么办？
    '.*prototype.*': dict(lr=lr,
                          LARS_exclude=True,
                          WD_exclude=True,
                          weight_decay=0)
})
optimizer_config = dict(grad_clip=dict(max_norm=3.0, norm_type=2))

# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
# yapf:enable
# You can register your own hooks like this
custom_hooks = [dict(type='MSNHook',
                     start_sharpen=0.25,
                     final_sharpen=0.25,
                     start_momentum=0.996,
                     final_momentum=1.0,
                     start_weight_decay=0.04,
                     final_weight_decay=0.4,
                     )]

# runtime settings
total_epochs = 800
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='MSN Pretrain',
                #id='2uwapqfv',
                #resume='allow',
                config=dict(
                    model=model,
                    rand_view_pipeline=rand_view_pipeline,
                    focal_view_pipeline=focal_view_pipeline,
                    epochs=runner['max_epochs'],
                    batch_size=data['samples_per_gpu'],
                    optimizer=optimizer,
                    optimizer_config=optimizer_config,
                    lr_config=lr_config,
                    data_settings=data,
                    ),
                )
            )
    ])