import cv2
import os
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def get_PSNR(img1, img2):
    # MSE = mean_squared_error(img1, img2)
    PSNR = peak_signal_noise_ratio(img1, img2)
    # print('MSE: ', MSE)
    # print('PSNR: ', PSNR)
    return PSNR

def get_SSIM(img1, img2):
    SSIM = structural_similarity(img1, img2, multichannel=True)
    # print('structural similarity: ', SSIM)
    return SSIM


##### Calculate PSNR and SSIM for images in folder #####
if __name__ == '__main__':

    # Ground truth folder and results folder
    GTs = 'G:\BACKUP\learning\programs\OracleRecognition\OracleAug\RCRN\JiaGu\Testing Set/test-gt'
    path_results = 'G:\BACKUP\learning\programs\OracleRecognition\OracleAug\RCRN\JiaGu\Testing Set/noise2same/Real_noise'
    # channel = 1

    SSIM_a = 0
    PSNR_a = 0
    num = 0

    GT = os.listdir(GTs)
    # GTpp = []
    # Targetpp = []
    for gt in GT:
        # Get pic names in output files
        name = gt.split(".")[0]
        result_path = path_results + '/' + name + '.png'
        gt_path = GTs + '/' + name + '.png'

        try:
            result_img = cv2.imread(result_path)
            GT_img = cv2.imread(gt_path)

            # if channel == 1:
            #     result_img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            #     GT_img = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            # elif channel == 3:
            #     result_img = cv2.imread(result_path)
            #     GT_img = cv2.imread(gt)

            dim = (GT_img.shape[1], GT_img.shape[0])
            # dim = (256, 256)
            # GT_img = cv2.resize(GT_img, dim)
            result_img = cv2.resize(result_img, dim)
            # _, result_img = cv2.threshold(result_img, 100, 255, cv2.THRESH_BINARY)

            pp = get_PSNR(GT_img, result_img)
            ss = get_SSIM(GT_img, result_img)

            # GTpp.append(GT_img)
            # Targetpp.append(result_img)

        except:
            print("error:-----------------------")
            print(gt)
            p = 0
            s = 0
            num -= 1

        SSIM_a += ss
        PSNR_a += pp
        num += 1

    # targetbbb = np.stack(Targetpp, axis=0)
    # GTbbb = np.stack(GTpp, axis=0)
    #
    # PSNR_b = get_PSNR(GTbbb, targetbbb)
    # SSIM_b = get_SSIM(GTbbb, targetbbb)

    print("PSNR: ", PSNR_a/num)
    print("SSIM: ", SSIM_a/num)
    # print("PSNR: ", PSNR_b)
    # print("SSIM: ", SSIM_b)