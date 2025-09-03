from types import SimpleNamespace

cfg= SimpleNamespace()
cfg.seed = 42

cfg.backbone = 'seresnext50_32x4d'
cfg.backbone2 = 'tf_efficientnet_b3_ns'

cfg.epochs = 100
cfg.batch_size = 1
cfg.batch_size_val = 1
cfg.lr = 2e-4
cfg.min_lr = 1e-6
cfg.weight_decay = 1e-2
cfg.num_workers = 4
cfg.num_classes = 6

cfg.early_stopping = {"patience": 50, "streak": 0}
cfg.folds = [0, 1, 2, 3, 4]

cfg.dataroot = './images/train'
cfg.testroot = './TestSetB'
cfg.csv_dir = './train_folds_zby_1.csv'
cfg.model_dir = './outputs/Seresnext101_v001'
cfg.result_dir = './result.txt'