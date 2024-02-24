_base_ = [
    './microplastic_detection_256x256.py',
    'mmseg::_base_/models/fcn_unet_s5-d16.py',
    'mmseg::_base_/default_runtime.py',
    # 'mmseg::_base_/schedules/schedule_20k.py'
]
train_dataloader = dict(dataset=dict(ann_file='train_EvalProtocol_SED_INTRA_INTER_ILE.txt'))
test_dataloader = dict(dataset=dict(ann_file='test_EvalProtocol_SED_INTRA_INTER_ILE.txt'))

custom_imports = dict(imports='mmseg.datasets.microplastic')
# model config overrides
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size, pad_val=0, seg_pad_val=0)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2,
                     out_channels=2,
                     ignore_index=0,
                     loss_decode=[
                         # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, use_sigmoid=True
                         #      #class_weight=[0.01, 0.99]
                         #      ),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, use_sigmoid=True, ignore_index=0)
                     ]
                     ),
    auxiliary_head=None,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(170, 170)),
    #test_cfg=dict(mode='whole'),
    )

# runtime config overrides
load_from = None  # todo add pretrained model
vis_backends = [dict(type='TensorboardVisBackend'), dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

# scheduler config overrides
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=4000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=200)
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)  # delete true ignore
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# other args from base config for this variable
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=50),  # log after k iterations
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200, save_best=['mIoU'], rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=30)  # batch = 1 so it will be every 50 images
)


