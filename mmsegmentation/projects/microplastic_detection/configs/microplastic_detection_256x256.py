dataset_type = 'MicroPlasticDataset'
data_root = 'data/processed/prepare_dataset_for_openmmseg2'
crop_size = (256, 256)
custom_imports = dict(imports='mmseg.datasets.transforms.custom_transforms')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='InvertBinaryLabels'),
    dict(
        type='RandomResize',
        scale=(640, 400),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCropForeground', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5),
         hue_delta=2),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 400), keep_ratio=True),  # gt do not need to be resized as it will
    # be resized to the network output
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='InvertBinaryLabels'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        ann_file='train_EvalProtocol_BENI_INTRA_ILE.txt',
        data_prefix=dict(img_path='data', seg_map_path='labels'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        ann_file='test_EvalProtocol_BENI_INTRA_ILE.txt',
        data_prefix=dict(img_path='data', seg_map_path='labels'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
