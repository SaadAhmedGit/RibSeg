import os
import numpy as np
from tqdm import tqdm


def evaluate(input_dir, second_stage_dir=None):  # binary_dir, binary=False):
    test_set = list(range(501, 661))
    for idx in [501, 507, 630]:
        test_set.remove(idx)

    dice_avg = 0
    single_rib_recall = np.zeros((24, 2))
    for idx in tqdm(test_set):
        name = f"RibFrac{idx}.npz"

        if second_stage_dir is None:
            res_data = np.load(os.path.join(input_dir, name))
            # res_data_binary = np.load(os.path.join(binary_dir, name))
            gt = res_data["gt"]

            pred = res_data["pred"]  # if not binary else res_data_binary['pred']
        else:
            binary = np.load(os.path.join(input_dir, name))
            second_stage = np.load(os.path.join(second_stage_dir, name))

            assert np.sum(binary["pred"] > 0) == second_stage["pred"].shape[0]
            binary_is_valid = (
                binary["pred"] == 0
            )  # NOTE: since filtering is done based on pred and not seg

            gt = np.concatenate([binary["gt"][binary_is_valid], second_stage["gt"]])
            pred = np.concatenate(
                [binary["pred"][binary_is_valid], second_stage["pred"]]
            )

        dice = np.zeros(24)
        num = 24
        for i in range(24):
            gt_, pred_ = gt.copy(), pred.copy()
            gt_[gt_ != i + 1] = 0
            gt_[gt_ == i + 1] = 1
            pred_[pred_ != i + 1] = 0
            pred_[pred_ == i + 1] = 1
            intersection = (gt_ * pred_).sum()
            union_pred = pred_.sum()
            union_gt = gt_.sum()
            if union_gt:
                dice_temp = (2 * intersection) / (union_pred + union_gt)
                dice[i] += dice_temp
                single_rib_recall[i][1] += 1
                # print("USING AVERAGE DICE")
                # single_rib_recall[i][0] += dice_temp
                if dice_temp > 0.7:
                    single_rib_recall[i][0] += 1
            else:
                num -= 1
        dice_avg += dice.sum() / num
    dice_avg /= len(test_set)
    print(dice_avg)
    recall1 = (single_rib_recall[1 - 1][0] + single_rib_recall[13 - 1][0]) / (
        (single_rib_recall[1 - 1][1] + single_rib_recall[13 - 1][1])
    )
    recall3 = (single_rib_recall[12 - 1][0] + single_rib_recall[24 - 1][0]) / (
        (single_rib_recall[12 - 1][1] + single_rib_recall[24 - 1][1])
    )
    a = (
        single_rib_recall[:, 0].sum()
        - single_rib_recall[1 - 1][0]
        - single_rib_recall[13 - 1][0]
        - single_rib_recall[12 - 1][0]
        - single_rib_recall[24 - 1][0]
    )
    b = (
        single_rib_recall[:, 1].sum()
        - single_rib_recall[1 - 1][1]
        - single_rib_recall[13 - 1][1]
        - single_rib_recall[12 - 1][1]
        - single_rib_recall[24 - 1][1]
    )
    recall2 = a / b
    recall4 = single_rib_recall[:, 0].sum() / single_rib_recall[:, 1].sum()
    print("average rib acc data:", recall4, recall1, recall2, recall3)


def main():
    print("pointcnn")
    evaluate("/data/adhinart/ribseg/outputs/pointcnn")

    print("pointcnn_two_stage")
    evaluate(
        "/data/adhinart/ribseg/outputs/pointcnn_binary",
        second_stage_dir="/data/adhinart/ribseg/outputs/pointcnn_binary_second_stage",
    )

    print("dgcnn")
    evaluate("/data/adhinart/ribseg/outputs/dgcnn")

    print("dgcnn_two_stage")
    evaluate(
        "/data/adhinart/ribseg/outputs/dgcnn_binary",
        second_stage_dir="/data/adhinart/ribseg/outputs/dgcnn_binary_second_stage",
    )
    print("pointnet1")
    evaluate("/data/adhinart/ribseg/outputs/pointnet1")

    print("pointnet1_two_stage")
    evaluate(
        "/data/adhinart/ribseg/outputs/pointnet1_binary",
        second_stage_dir="/data/adhinart/ribseg/outputs/pointnet1_binary_second_stage",
    )

    print("pointnet2")
    evaluate("/data/adhinart/ribseg/outputs/pointnet2")
    print("pointnet2_two_stage")
    evaluate(
        "/data/adhinart/ribseg/outputs/pointnet2_binary",
        second_stage_dir="/data/adhinart/ribseg/outputs/pointnet2_binary_second_stage",
    )
    print("pointnet1_2048")
    evaluate("/data/adhinart/ribseg/outputs/pointnet1_2048")

    print("pointnet1_2048_two_stage")
    evaluate(
        "/data/adhinart/ribseg/outputs/pointnet1_2048_binary",
        second_stage_dir="/data/adhinart/ribseg/outputs/pointnet1_2048_binary_second_stage",
    )

    print("pointnet2_2048")
    evaluate("/data/adhinart/ribseg/outputs/pointnet2_2048")
    print("pointnet2_2048_two_stage")
    evaluate(
        "/data/adhinart/ribseg/outputs/pointnet2_2048_binary",
        second_stage_dir="/data/adhinart/ribseg/outputs/pointnet2_2048_binary_second_stage",
    )

    print("nnunet")
    evaluate("/data/adhinart/ribseg/outputs/nnunet_3d_fullres")


if __name__ == "__main__":
    main()
