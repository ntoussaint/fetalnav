import os
import numpy as np
import torch
import torchvision.transforms as torchtransforms

from .models.spn_models import vgg16_sp
from .transforms import tensor_transforms as tensortransforms


datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
modelfile = os.path.join(datadir, 'vgg16_sp_miccai2018.pth.tar')
classes = ['Abdomen', 'Background', 'Head', 'Limbs', 'Placenta', 'Spine', 'Thorax']


# transform a numpy array into a torch tensor
totensor = torchtransforms.ToTensor()
# crop to an aspect ratio
crop = tensortransforms.CropToRatio(outputaspect=1.5)
# padd the responses according the the initial crop:
padd = tensortransforms.PaddToRatio(outputaspect=1.5)
# resize image to fixed size
resize = tensortransforms.Resize(size=[224, 224], interp='bilinear')
# rescale tensor to  interval
rescale = tensortransforms.Rescale(interval=(0, 1))
# rescale responses so they are between 0 and 255:
descale = tensortransforms.Rescale(interval=(0, 255))

transform = torchtransforms.Compose(
                [totensor,
                 crop,
                 resize,
                 rescale])

vgg = vgg16_sp(len(classes), num_maps=512, batch_norm=True, in_channels=1)

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
    scores = torch.nn.Softmax(dim=1)(model(in_var)).data.cpu().squeeze()
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
        # ensure the size is enough
        if size[0] < 50 or size[1] < 50:
            print ("[fetalnav] - WARNING: image size is invalid: {}".format(size))
            return defaultret, None
        # reshape to get a 3 dimensional image with 1 as z
        npimage = np.reshape(image_cpp, [size[0], size[1], 1])
        # create the torch variable valid for model input
        torchinput = Variable(transform(npimage).unsqueeze(0))

        # predict the class
        output, responses = generate_outputs(vgg, torchinput)

        responses = descale(responses)
        responses = padd(responses)
        # resize the responses to the initial input size:
        desize = tensortransforms.Resize(size=size, interp='bilinear')
        responses = desize(responses)
        # impose dtype to uint8
        responses = responses.cpu().numpy().squeeze().astype(np.uint8)

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
