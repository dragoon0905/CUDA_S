def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
def MIOU(pred,label):
    hist = np.zeros((19, 19))
    hist += fast_hist(label.flatten(), pred.flatten(), 19)
    mIoUs = per_class_iu(hist)
    if True:
        for ind_class in range(18):
            print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU (GTA5): ' + str(round(np.nanmean(mIoUs[:18]) * 100, 2)))



def _mask_transform2(pred):
    label_copy = 18 * np.ones(pred.shape, dtype=np.float32)
    for k, v in id_to_miouid.items():
        label_copy[label_copy == k] = v
    return label_copy



id_to_miouid = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
}