import numpy as np
import nibabel as nib
import skimage
from skimage import morphology as morph
from skimage.measure import label,regionprops
import time,os
from tqdm import tqdm
from collections import Counter
import operator
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline

source_data_dir = './data/ribfrac/ribfrac-val-images/'
Label_dir = './data/RibSeg/nii/'

dis_c2_point_dir = './inference_res/point/'
dis_c2_label_dir = './inference_res/label/'


### add .npy when loading point/label
name_list = tqdm([x for x in os.listdir(source_data_dir)])
## source image
total_dice = 0
rib_recall = np.zeros((24, 2))
recall = np.zeros(24)
num = 0
for ct in name_list:
    num += 1

    s_i = nib.load(source_data_dir + ct)
    s_i = s_i.get_fdata()
    s_i[s_i != 0] = 1
    s_i = s_i.astype('int8')

    loc = np.load(dis_c2_point_dir + 'val' + ct[:-13]+'.npy')
    label = np.load(dis_c2_label_dir + 'val' + ct[:-13]+'.npy')

    mask_rd = np.zeros(s_i.shape)
    mask_res = np.zeros(s_i.shape)

    for index in loc:
        x, y, z = index[0], index[1], index[2]
        mask_rd[x][y][z] = 1
    for i in range(loc.shape[0]):
        index = loc[i]
        x, y, z = index[0], index[1], index[2]
        mask_res[x][y][z] = label[i]

    lmage_array = sitk.GetImageFromArray(mask_res.astype('int8'))
    # closed = sitk.BinaryMorphologicalClosing(lmage_array,15,sitk.sitkBall)
    # You should try different parameters for better results.
    dilated = sitk.BinaryDilate(lmage_array, (3,3,3), sitk.sitkBall)
    # Eroded = sitk.BinaryErode(dilated,12,sitk.sitkBall)
    # holesfilled = sitk.BinaryFillhole(dilated,fullyConnected=True)
    holesfilled = sitk.GetArrayFromImage(dilated)

    res = np.multiply(s_i, holesfilled)

    res1 = skimage.measure.label(res, connectivity=1)
    rib_p = regionprops(res1)

    rib_p.sort(key=lambda x: x.area, reverse=True)

    im = np.in1d(res1, [x.label for x in rib_p[:24]]).reshape(res1.shape)

    im = im.astype('int8')

    ## dice
    s = nib.load(Label_dir + ct[:-12]+'label.nii.gz')
    s = s.get_fdata()
    s = s.astype('int8')

    intersection = np.multiply(s, im)
    insec = np.argwhere(intersection != 0).shape[0]
    s_i = np.argwhere(s != 0).shape[0]
    i_i = np.argwhere(im != 0).shape[0]

    dice = 1 - (2 * insec + 1) / (s_i + i_i + 1)

    total_dice += dice

    ## recall

    for i in range(24):
        loc = np.argwhere(s == i + 1)
        loc_num = loc.shape[0]
        rib_count = 0

        if loc_num != 0:
            rib_recall[i][1] += 1
            for index in loc:
                x, y, z = index[0], index[1], index[2]
                if im[x][y][z] != 0:
                    rib_count += 1
            if rib_count >= loc_num * 0.5:
                rib_recall[i][0] += 1

for x in range(24):
    recall[x] = rib_recall[x][0] / rib_recall[x][1]
ave_dice = total_dice / num

recall1 = (rib_recall[1 - 1][0] + rib_recall[13 - 1][0]) / ((rib_recall[1 - 1][1] + rib_recall[13 - 1][1]))
recall3 = (rib_recall[12 - 1][0] + rib_recall[24 - 1][0]) / ((rib_recall[12 - 1][1] + rib_recall[24 - 1][1]))
a = rib_recall[:, 0].sum() - rib_recall[1 - 1][0] - rib_recall[13 - 1][0] - rib_recall[12 - 1][0] - rib_recall[24 - 1][
    0]
b = rib_recall[:, 1].sum() - rib_recall[1 - 1][1] - rib_recall[13 - 1][1] - rib_recall[12 - 1][1] - rib_recall[24 - 1][
    1]
recall2 = (a / b)
recall4 = (rib_recall[:, 0].sum() / rib_recall[:, 1].sum())
print('average rib recall data:', recall1, recall2, recall3, recall4)
print('dice:', ave_dice)
