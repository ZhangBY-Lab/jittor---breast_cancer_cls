import jittor as jt
import numpy as np
import os
from _cfg import cfg
from _model import CancerModel
from _datasets import ImageFolder, build_transform
from _utils import device_usr, create_dir, initialize_random_states


def test(model_list, data_loader, save_txt):
    for m in model_list:
        m.eval()

    # 用来放每张图的投票
    vote_dict = {}

    print("start testing ...")

    for one_batch in data_loader:
        imgs, names = one_batch  # imgs: (B,C,H,W)

        aug_list = [imgs,
                    imgs[:, :, ::-1, :],  # 上下翻
                    imgs[:, :, :, ::-1]]  # 左右翻

        for cur_img in aug_list:
            for m in model_list:
                with jt.no_grad():
                    out = m(cur_img).numpy()
                    pre_label = np.argmax(out, axis=1)

                for k in range(len(names)):
                    fname = names[k]
                    lab = int(pre_label[k])
                    if fname not in vote_dict:
                        vote_dict[fname] = []
                    vote_dict[fname].append(lab)

    create_dir(os.path.dirname(save_txt))
    fp = open(save_txt, 'w', encoding='utf-8')

    for fname in vote_dict:
        arr = vote_dict[fname]
        # 求众数
        cnt = np.bincount(arr)
        final_lab = int(np.argmax(cnt))
        fp.write(f"{fname} {final_lab}\n")

    fp.close()
    print("done.")


if __name__ == '__main__':
    initialize_random_states(42)
    transform = build_transform()

    test_loader = ImageFolder(
        root=cfg.testroot,
        transform=transform,
        batch_size=cfg.batch_size_val,
        num_workers=cfg.num_workers,
        num_classes=cfg.num_classes,
        shuffle=False,
        mode="val"
    )
    model_weights = ["./outputs/zby_final/checkpoint1.pkl", "./outputs/zby_final/checkpoint2.pkl",
                     "./outputs/zby_final/checkpoint3.pkl"]  # 21 0.92 35 0.9155 16 0.918

    models = []
    for weight in model_weights:
        print(weight)
        model = CancerModel(
            model_name=cfg.backbone,
            model_name2=cfg.backbone2,
            num_classes=6,
            pretrain=True,
            pooling_type='gem',
            dropout_rate=0.4
        )

        model.load(weight)
        models.append(model)
    test(models, test_loader, cfg.result_dir)