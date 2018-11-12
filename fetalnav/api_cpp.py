import os
import numpy as np
import torch
from torchvision import transforms as torchtransforms
from PIL import Image


from .models.spn_models import vgg16_sp, vgg13_sp, resnet18_sp
from .transforms import tensor_transforms as tensortransforms

# datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
# modelfile = os.path.join(datadir, 'vgg16_sp_miccai2018.pth.tar')
modelfile = '/home/nt08/Projects/fetalnav-github/experiments/logs/cs=pol-m=vgg13_bn-lr=0.05-bs=7-spn=1-aspect=1.5/checkpoint_17.pth.tar'
classes = ['Abdomen', 'Background', 'Head', 'Limbs', 'Placenta', 'Spine', 'Thorax']

desiredaspect = 1.5
desiredsize = [224] * 2
cropping = True

# transform a numpy array into a torch tensor
totensor = torchtransforms.ToTensor()
# crop to an aspect ratio
crop = tensortransforms.CropToRatio(outputaspect=desiredaspect)
# padd the responses according the the initial crop:
padd = tensortransforms.PaddToRatio(outputaspect=desiredaspect)
# resize image to fixed size
resize = tensortransforms.Resize(size=desiredsize, interp='bilinear')
# rescale tensor to  interval
rescale = tensortransforms.Rescale(interval=(0, 1))
# rescale responses so they are between 0 and 255:
descale = tensortransforms.Rescale(interval=(0, 255))

if cropping:
    transform = torchtransforms.Compose(
                    [totensor,
                     crop,
                     resize,
                     rescale])
else:
    transform = torchtransforms.Compose(
                    [totensor,
                     resize,
                     rescale])

# vgg = resnet18_sp(len(classes), num_maps=512, in_channels=1)
vgg = vgg13_sp(len(classes), num_maps=512, batch_norm=True, in_channels=1)

print('loading network: {}'.format(modelfile))
checkpoint = torch.load(modelfile)
vgg.load_state_dict(checkpoint['state_dict'])
if torch.cuda.is_available():
    vgg = torch.nn.DataParallel(vgg).cuda()
else:
    print('[fetalnav] - WARNING: detection works best using GPU... ')
vgg.eval()


def getlabels():
    return classes


def generate_outputs(model, in_var):

    from spn import hook_spn
    from torch.nn import functional as F

    if in_var.ndimension() == 3:
        input = in_var.unsqueeze(0)
    assert in_var.size(0) == 1, 'Batch processing is currently not supported'
    # enable spn inference mode
    model = hook_spn(model)
    # predict scores
    scores = torch.nn.Sigmoid()(model(in_var)).data.cpu().squeeze()
    # instantiate maps
    maps = F.upsample(model.class_response_maps, size=(in_var.size(2), in_var.size(3)), mode='bilinear').data
    return scores.numpy(), maps


def getprediction(image_cpp, verbose=False):

    from torch.autograd import Variable

    # define a default return value
    defaultret = np.array([0.0] * len(classes), dtype=np.float32)

    try:
        # make sure the entry is numpy array of type float32
        image_cpp = np.array(image_cpp, dtype=np.float32)
        # check image size
        size = image_cpp.shape
        inputaspect = size[0]/size[1]
        # ensure the size is enough
        if size[0] < 50 or size[1] < 50:
            print("[fetalnav] - WARNING: image size is invalid: {}".format(size))
            return defaultret, None
        # reshape to get a 3 dimensional image with 1 as z
        npimage = np.reshape(image_cpp, [size[0], size[1], 1])
        Image.fromarray(npimage.squeeze().astype(np.uint8)).save("/tmp/npimage.png")
        # create the torch variable valid for model input
        torchinput = Variable(transform(npimage).unsqueeze(0))

        # Image.fromarray(descale(transform(npimage)).squeeze().numpy().astype(np.uint8)).save("/tmp/torchinput.png")

        # predict the class
        output, responses = generate_outputs(vgg, torchinput)

        # responses = descale(responses)
        responses = torch.stack([descale(responses[i]) for i in range(responses.size(0))])

        if cropping:
            newsize = list(size)
            if inputaspect > desiredaspect:
                newsize[0] = int(size[0] * desiredaspect / inputaspect)
            else:
                newsize[1] = int(size[1] * inputaspect / desiredaspect)

            # resize the responses to the initial input size:
            desize = tensortransforms.Resize(size=newsize, interp='bilinear')
            responses = desize(responses)

            padd = tensortransforms.PaddToRatio(outputaspect=inputaspect)
            responses = padd(responses)
        else:
            desize = tensortransforms.Resize(size=size, interp='bilinear')
            responses = desize(responses)

        responses = responses.cpu().numpy().squeeze().astype(np.uint8)

        # Image.fromarray(responses[0]).save("/tmp/abdoresponse.png")
        # Image.fromarray(responses[3]).save("/tmp/limbsresponse.png")

        # optionally save the highest response map
        if verbose is True:
            print('fetalnav: [ %04.2f, %04.2f, %04.2f, %04.2f, %04.2f, %04.2f ] - \t%s' % (output[0], output[1], output[2], output[3], output[4], output[5], classes[np.argmax(output)]))

        # return output
        return output, responses

    except Exception as inst:
        print ('[fetalnav] - WARNING: Exception detected')
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
        return defaultret, None
