_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    './car_damage_detection_1cat_1image.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        frozen_stages=0),
    bbox_head=dict(
        num_classes=1),  # 80),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.05),
        max_per_img=100))


### schedule
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)  # lr = 0.01

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[10000])
runner = dict(type='EpochBasedRunner', max_epochs=50)

work_dir = '/home/dalya/PycharmProjects/mmdet/mmdetection/results/1cat_1image_frozen0'

### runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/dalya/PycharmProjects/mmdet/mmdetection/results/overfit_1cat_exp2/epoch_15.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
