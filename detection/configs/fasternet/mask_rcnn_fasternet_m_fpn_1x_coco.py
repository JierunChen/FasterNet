_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_m',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../model_ckpt/fasternet_m-epoch=291-val_acc1=82.9620.pth',
            ),
        # init_cfg=None,
        ),
    neck=dict(in_channels=[144, 288, 576, 1152]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
