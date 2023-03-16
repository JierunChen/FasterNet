_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_s',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../model_ckpt/fasternet_s-epoch=299-val_acc1=81.2840.pth',
            ),
        # init_cfg=None,
        ),
    neck=dict(in_channels=[128, 256, 512, 1024]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
