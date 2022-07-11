import torch
import time
import argparse
from torch.autograd import Variable
from torchvision.utils import make_grid

from models.model_CharFormer import CharFormer
from datasets import *
from util.TestMetrics import get_PSNR, get_SSIM
# from datasets import skeletonPrepare

########################################
#####      Tools Definitions       #####
########################################

##### Load weights of the model #####
def loadModel(model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


##### Concat input and target pics for visual comparison #####
def concat(imgA, imgB):
    size1, size2 = imgA.size, imgB.size

    joint = Image.new(opt.color_set, (size1[0] + size2[0], size1[1]))
    loc1, loc2 = (0, 0), (size1[0], 0)
    joint.paste(imgA, loc1)
    joint.paste(imgB, loc2)
    # joint.show()
    return joint


##### Image pre-processing #####
def AdditionalProcess(pic):
    # If input pic is the combination of input and target pics
    if opt.IsPre == True:
        width, height = pic.size
        pic = pic.crop((width / 2, 0, width, height))
    Ori_width, Ori_height = pic.size

    # If there is additional processing on input pics
    # pic = skeletonPrepare(pic)
    return pic, Ori_width, Ori_height

##### Save a given Tensor into an image file #####
def Get_tensor(tensor, nrow=8, padding=2,
               normalize=False, irange=None, scale_each=False, pad_value=0):

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, irange=irange, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


########################################
#####       Image Processing       #####
########################################

if __name__ == '__main__':

    ##### arguments settings #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/Dataset/test", help="path of input pics")
    parser.add_argument("--store_path", type=str, default="data/Dataset/predict-results/",
                        help="path for stroing the output pics")
    parser.add_argument("--model_path", type=str, default="saved_models/Dataset2/CharFormer_20.pth",
                        help="path for loading the trained models")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--color_set", type=str, default="RGB", help="number of image color set, RGB or L ")
    parser.add_argument("--IsPre", type=bool, default=True, help="If need to make pre-processing on input pics")
    parser.add_argument("--OutputComparisons", type=bool, default=True,
                        help="If need to make pre-processing on input pics")
    parser.add_argument("--dataset_name", type=str, default="Oracles2", help="name of the dataset")
    opt = parser.parse_args()

    ##### Create store_path #####
    os.makedirs(opt.store_path, exist_ok=True)

    ##### Initialize CUDA and Tensor #####
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ##### data transformations #####
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]
    transform = transforms.Compose(transforms_)


    ##### Process input pics #####
    pics = os.listdir(opt.input_path)
    for pic in pics:
        # Get pic name and set names of output files
        name = pic.split(".")[0]
        output_path = os.path.join(opt.store_path, name) + ".png"
        output_path_compare = os.path.join(opt.store_path, name) + "_compare.jpeg"

        # Read pic by PIL and make Pre-preocessing before feeding the input pic
        image = Image.open(os.path.join(opt.input_path, pic)).convert(opt.color_set)
        Ori_image, Ori_width, Ori_height = AdditionalProcess(image)
        image = transform(Ori_image)

        # Resize the pic in to (batch, channel, width, height)
        image = image.resize(1, opt.channels, opt.img_width, opt.img_height)
        image = Variable(image.type(Tensor))

        # Load the weights of model
        generator = loadModel(CharFormer())

        # Calculate inference period of the model for single image, and get the output image
        start = time.time()
        output = generator(image)
        end = time.time()
        print("processing time: ", (end - start) * 1000, "ms")

        # Obtain output image and convert to PIL format, then resize the image into the original size
        Output_img = Get_tensor(output.data, normalize=True)
        Output_img = Output_img.resize((Ori_width, Ori_height))
        Output_img.save(output_path)

        # Save the combination of input and target pics for visual comparison
        if opt.OutputComparisons == True:
            Comparison_img = concat(Output_img, Ori_image)
            Comparison_img.save(output_path_compare)

