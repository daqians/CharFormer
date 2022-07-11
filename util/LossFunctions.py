import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from skimage import morphology

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

##### Get tensor and transform it into PIL format #####
def Get_tensor(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


##### Pixel-wise loss #####
class PixelwiseLoss(nn.Module):
    def forward(self, inputs, targets):
        return F.smooth_l1_loss(inputs, targets)

##### MSE loss #####
# def perceptual_loss(x, y):
#     F.mse_loss(x, y)



########################################
#####    Defined loss functions    #####
########################################

##### Perceptual loss (VGG loss) #####
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # input = input.cuda()
        # target = target.cuda()
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            if torch.cuda.is_available():
                block.cuda()

            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)

        return loss


##### Skeleton loss #####
def SK_loss(input,target):
    def skeletonize(image):
        # improved skeletonization function
        image = np.uint8(image)
        ret, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary)
        skeleton = skeleton0.astype(np.uint8) * 255
        return skeleton

    input = Get_tensor(input.data, normalize=True)
    target = Get_tensor(target.data, normalize=True)
    input = cv2.cvtColor(np.array(input), cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(np.array(target), cv2.COLOR_BGR2GRAY)

    input_sk = skeletonize(input)
    target_sk = skeletonize(target)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),]
    transform = transforms.Compose(transforms_)

    input_sk = transform(input_sk)
    target_sk = transform(target_sk)


    return F.mse_loss(input_sk, target_sk)








