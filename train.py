##### External Interface #####
import argparse
import time
import datetime
import sys
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid

##### Internal Interface #####
from util.LossFunctions import VGGPerceptualLoss, SK_loss
from models.model_CharFormer import CharFormer
from datasets import *
from util.TestMetrics import get_PSNR, get_SSIM

##### Optional Tools #####
from DNN_printer import DNN_printer
import wandb
import wmi


def TrainModel(opt):
    ##### Make necessary directories #####
    os.makedirs("validImages/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    ##### Determine CUDA #####
    cuda = True if torch.cuda.is_available() else False
    # device = torch.device("cuda" if cuda else "cpu")

    ##### Initialize models: generator and discriminator #####
    model = CharFormer(dim=16, stages=opt.stages, depth_RSAB = opt.depth_RSAB, depth_GSNB = opt.depth_GSNB, dim_head=64, heads=8)
    if cuda:
        model = model.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        model.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))



    ##### Print Model Size #####
    DNN_printer(model, (opt.channels, opt.img_height, opt.img_width), opt.batch_size)

    ##### Define Optimizers #####
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    ##### Configure Dataloaders #####
    # data transformations
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]

    # training data
    dataloader = DataLoader(
        ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, file_set="separate",
                     root2="data/%s/target" % opt.dataset_name),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # validating data
    val_dataloader = DataLoader(
        ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, mode="val", file_set="separate",
                     root2="data/%s/target" % opt.dataset_name),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    ########################################
    #####      Parameter Setting       #####
    ########################################

    ##### Loss functions #####
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss()
    if cuda:
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    ##### Loss weight of L1 pixel-wise loss between translated image and real image #####
    lambda_pixel = 100
    lambda_vgg = 0.05
    lamdba_sk = 100

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ########################################
    #####      Tools Definitions       #####
    ########################################

    ##### calculate the evaluation metrics on validation set #####
    def output_metrics():
        imgs = next(iter(val_dataloader))
        input = Variable(imgs["B"].type(Tensor))
        target = Variable(imgs["A"].type(Tensor))
        output, features_ = model(input)

        ndarr_target = make_grid(target.data).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                    torch.uint8).numpy()
        ndarr_output = make_grid(output.data).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                    torch.uint8).numpy()

        PSNR = get_PSNR(ndarr_target, ndarr_output)
        SSIM = get_SSIM(ndarr_target, ndarr_output)

        return PSNR, SSIM

    ##### Saves a generated sample from the validation set #####
    def sample_images(batches_done):
        imgs = next(iter(val_dataloader))
        input = Variable(imgs["B"].type(Tensor))
        target = Variable(imgs["A"].type(Tensor))
        output, features_ = model(input)
        # img_sample = torch.cat((input.data, features_.data, output.data, target.data), 2)
        img_sample = torch.cat((input.data, output.data, target.data), 2)
        save_image(img_sample, "validImages/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

    ##### Monitor the temperature of the device to avoid overheating #####
    def avg(value_list):
        num = 0
        length = len(value_list)
        for val in value_list:
            num += val
        return num / length

    def temp_monitor():
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()
        cpu_temps = []
        gpu_temp = 0
        for sensor in sensors:
            if sensor.SensorType == u'Temperature' and not 'GPU' in sensor.Name:
                cpu_temps += [float(sensor.Value)]
            elif sensor.SensorType == u'Temperature' and 'GPU' in sensor.Name:
                gpu_temp = sensor.Value

        # print("Avg CPU: {}".format(avg(cpu_temps)))
        # print("GPU: {}".format(gpu_temp))
        if avg(cpu_temps) > opt.device_temperature or gpu_temp > opt.device_temperature:
            print("Avg CPU: {}".format(avg(cpu_temps)))
            print("GPU: {}".format(gpu_temp))
            print("sleeping 30s")
            time.sleep(30)
        return

    ########################################
    #####           Training           #####
    ########################################

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            ##### Model Inputs: input pics and target pics #####
            input = Variable(batch["B"].type(Tensor))
            target = Variable(batch["A"].type(Tensor))
            target_sk = Variable(batch["F"].type(Tensor))

            # ------------------
            #  Train Generators
            # ------------------

            optimizer.zero_grad()

            ##### Generator Losses #####
            output,feature_ = model(input)
            loss_pixel = criterion_pixelwise(output, target)
            loss_VGG = criterion_VGG(output, target)

            loss_skeleton = criterion_pixelwise(feature_, target_sk)
            loss_VGG_sk = criterion_VGG(feature_, target_sk)

            ##### Total Generator Loss #####
            loss_total = lambda_pixel * loss_pixel + lambda_vgg * loss_VGG + lamdba_sk * loss_skeleton + loss_VGG_sk

            loss_total.backward()
            optimizer.step()



            # --------------
            #  Log Progress
            # --------------

            ##### Approximate finishing time #####
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            ##### Print log #####
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f,  VGG: %f] EstimateTime: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_total.item(),
                    loss_pixel.item(),
                    loss_VGG.item(),
                    time_left,
                )
            )

            ##### Save samples at intervals #####
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)
                temp_monitor()

        ##### Save model checkpoints #####
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       "saved_models/%s/%s_%d.pth" % (opt.dataset_name, opt.model_name, epoch))


        ##### Optional logs in each epoch (by wandb) #####
        PSNR, SSIM = output_metrics()
        wandb.log({
            "loss": loss_total.item(),
            "%s_PSNR_%s" % (opt.dataset_name, "Generator"): PSNR,
            "%s_SSIM_%s" % (opt.dataset_name, "Generator"): SSIM,
        })
        wandb.watch(model)


if __name__ == '__main__':
    ##### arguments settings #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=202, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="Dataset-npy", help="name of the dataset")
    parser.add_argument("--model_name", type=str, default="CharFormer", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--stages", type=int, default=4, help="stages of U net")
    parser.add_argument("--depth_RSAB", type=int, default=3, help="number of transformer per RSAB")
    parser.add_argument("--depth_GSNB", type=int, default=3, help="number of Conv2d per GSNB")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500,
                        help="interval between sampling of validImages from training")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
    parser.add_argument("--device_temperature", type=int, default=85,
                        help="Set maximum temperature of CPU and GPU for device safety")
    opt = parser.parse_args()

    ##### wandb initializing (Optional) #####
    wandb.login()
    wandb.init(project="CharFormer", entity="xxx", name="%s_1" % opt.model_name)
    # wandb.config.name = opt.model_name
    # wandb.config = {
    #     "name": opt.model_name,
    #     "learning_rate": 0.001,
    #     "epochs": 100,
    #     "batch_size": 128,
    # }

    ##### Run training process #####
    TrainModel(opt)
