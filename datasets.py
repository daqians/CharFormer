import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
from skimage import morphology
import time

class ImageDataset(Dataset):
    def __init__(self, root1, root2 = None, transforms_=None, mode="train", file_set ="separate", color_set ="RGB"):
        """
        root1: the input pics
        root2: the target pics that we want to preceed
        transforms_: the transform functions of each pic
        mode: results different ML mode for data selection, common options "train", "val"
        file_set: depends on two different resources for input pic:
          "combined" refers to the combination of input and target pic;
          "separate" refers to separated address of input and target pic
        color_set: color set of an image, "RGB" or "L"
        """

        self.file_set = file_set
        self.transform = transforms.Compose(transforms_)
        self.color_set = color_set

        if self.file_set == "combined":
            self.files = sorted(glob.glob(os.path.join(root1, mode) + "/*.*"))

        elif self.file_set == "separate":
            self.files = []
            for pics in sorted(glob.glob(os.path.join(root1, mode) + "/*.*")):
                pic_input = os.path.join(root1, mode, pics.split("\\")[-1])
                pic_target = os.path.join(root2, pics.split("\\")[-1])
                if os.path.exists(pic_target):
                    self.files.append((pic_input,pic_target))

    def __getitem__(self, index):
        # process the combination of target and input pic
        if self.file_set == "combined":
            img = Image.open(self.files[index % len(self.files)]).convert(self.color_set)
            w, h = img.size

            # separate the input and target pic from the pic combination
            img_A = img.crop((0, 0, w / 2, h))
            img_B = img.crop((w / 2, 0, w, h))

        # read two pics separately
        elif self.file_set == "separate":
            (input, target) = self.files[index % len(self.files)]
            # "L" for grayscale images, "RGB" for color images
            # img_A = Image.open(target).convert(self.color_set)
            # img_B = Image.open(input).convert(self.color_set)

            img_A = Image.fromarray(np.load(target))
            img_B = Image.fromarray(np.load(input))

        # reverse a pic for data augmentation
        if self.color_set == "RGB" and np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        if self.color_set == "L" and np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1], "L")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1], "L")

        sk = Image.fromarray(skeletonize(np.array(img_A.convert("1"))))

        # precess image transform functions
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        sk = self.transform(sk)

        return {"A": img_A, "B": img_B, "F": sk}

    def __len__(self):
        return len(self.files)

def skeletonize(image):
    # improved skeletonization function
    image = np.uint8(image)
    ret, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton
