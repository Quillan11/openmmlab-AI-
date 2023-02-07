_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(num_classes=5,
              topk=(1,3)),
)
dataset_type = 'ImageNet'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        data_prefix='/HOME/scz4253/run/mmclassification/mmclassification/data/flower/train',
        ann_file = '/HOME/scz4253/run/mmclassification/mmclassification/data/flower/train.txt',
        classes = '/HOME/scz4253/run/mmclassification/mmclassification/data/flower/classes.txt'),
    val=dict(
        data_prefix='/HOME/scz4253/run/mmclassification/mmclassification/data/flower/val',
        ann_file='/HOME/scz4253/run/mmclassification/mmclassification/data/flower/val.txt',
        classes = '/HOME/scz4253/run/mmclassification/mmclassification/data/flower/classes.txt'),
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

#预训练模型
load_from = '/HOME/scz4253/run/mmclassification/mmclassification/checkpoints/' \
            'resnet18_batch256_imagenet_20200708-34ab8f90.pth'
