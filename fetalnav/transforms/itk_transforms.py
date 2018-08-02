# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
Relevant Transforms of 2D/3D Biomedical Images using itk (sitk) images.
Transforms
1. ITK image to another ITK image
2. ITK image to pytorch tensors
3. Pytorch tensors to ITK images
"""

# Authors:
# Bishesh Khanal <bisheshkh@gmail.com>
# 	  King's College London, UK
#	  Department of Computing, Imperial College London, UK
# Nicolas Toussaint <nicolas.toussaint@gmail.com>
# 	  King's College London, UK

import SimpleITK as sitk
import math
import torch
import numpy as np


class ToNumpy(object):
    """Convert an itkImage to a ``numpy.ndarray``.
    Converts a itkImage (W x H x D) or numpy.ndarray (D x H X W)
    NOTA: itkImage ordering is different than of numpy and pytorch.
    """

    def __init__(self, outputtype=None):
        self.outputtype = outputtype

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : itkImages in SimpleITK format
            Images to be converted to numpy.
        Returns
        -------
        Numpy nd arrays
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            outputs.append(self._tonumpy(_input,self.outputtype))
        return outputs if idx > 0 else outputs[0]

    def _tonumpy(self, input, outputtype):
        ret = None
        if isinstance(input, sitk.SimpleITK.Image):
            # Extract the numpy nparray from the ITK image
            narray = sitk.GetArrayFromImage(input);
            # The image is now stored as (y,x), transpose it
            ret = np.transpose(narray, [1,2,0])
        elif isinstance(input, np.array):
            # if the input is already numpy, assume it is in the right order
            ret = input

        # overwrite output type if requested
        if self.outputtype is not None:
            ret = ret.astype(outputtype)

        return ret



class ToTensor(object):
    """Convert an itkImage or ``numpy.ndarray`` to tensor.
    Converts a itkImage (W x H x D) or numpy.ndarray (D x H X W) to a
    torch.FloatTensor of shape (D X H X W).
    i.e. itkImage ordering is different than of numpy and pytorch.
    """

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : itkImages or numpy.ndarrays
            Images to be converted to Tensor.
        Returns
        -------
        Tensors
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            _input_is_numpy = False
            if isinstance(_input, sitk.SimpleITK.Image):
                # Get numpy array (is a deep copy!)
                _input = sitk.GetArrayFromImage(_input)
                _input_is_numpy = True
            #print(f'input or converted numpy array type: {_input.dtype}')
            _input = torch.from_numpy(_input.astype(np.double))
            #_input = torch.from_numpy(_input)
            if _input_is_numpy:
                _input = _input.permute(2,1,0) #Change size from
            # float for backward compatibility ?
            outputs.append(_input.float())
        return outputs if idx > 0 else outputs[0]


class ToITKImage(object):

    def __init__(self, ref_img=None, itk_infos=None):
        """
        Converts one or more torch tensors or numpy ndarrays of shape D x H x W to itk image
        of shape W x H x D
        Takes metadata from ref_img or itk_infos if given otherwise uses default itk metadata.

        Arguments
        ---------
        ref_img : itkImage
            Reference image from which to take all meta information.
            Supports only one ref image.
        itk_infos : dictionary or a sequence (list or tuple) of dictionaries
            each dictionary with following keys:
            origin, spacing and direction

        """
        _l = [x for x in [ref_img, itk_infos] if x is not None]
        assert len(_l) <= 1, 'At most one of ref_img, itk_infos can be not none'
        self.ref_img = ref_img
        self.itk_infos = itk_infos

    def __call__(self, *inputs):
        outputs = []
        if not isinstance(self.itk_infos, (list, tuple)):
            self.itk_infos = (self.itk_infos,)*len(inputs)
        assert len(inputs) == len(self.itk_infos), 'num of inputs and itk_infos do not match'
        for idx, _input in enumerate(inputs):
            output_curr = self._toITK_image(_input, self.ref_img, self.itk_infos[idx])
            outputs.append(output_curr)
        return outputs if idx > 0 else outputs[0]

    def _toITK_image(self, tensor, ref_img=None, itk_info=None):
        """
        Arguments
        ---------
        tensor : Tensor or numpy.ndarray
            Tensor/array to be converted to ITK image.
            The ordering of input tensor is in z,y,x which will be converted
            to x,y,z.
        ref_img : itkImage
            Copy origin, spacing and direction from this image
        itk_info : dictionary with origin, spacing and direction as keys
            Overwrite info take from ref_img if not None

        Returns
        -------
            ITK image
        """

        if not isinstance(tensor, np.ndarray):
            tensor=tensor.permute(2,1,0)
            tensor = (tensor.cpu()).numpy()
        else:
            tensor = tensor.transpose(2,1,0) #numpy version of permute!

        itk_img = sitk.GetImageFromArray(tensor)
        if ref_img is not None:
            itk_img.CopyInformation(ref_img)
        if itk_info is not None:
            #print(f'itk_info: {itk_info}')
            itk_img.SetDirection(itk_info['direction'])
            itk_img.SetOrigin(itk_info['origin'])
            itk_img.SetSpacing(itk_info['spacing'])
        return itk_img


class Resample(object):

    def __init__(self, new_spacing=None, new_size=None, interp = 'linear'):
        """
        Resample an ITK image to either:
        a desired voxel spacing in mm given by [spXmm, spYmm, spZmm] or,
        a desired size [x, y, z]

        Arguments
        ---------
        `new_spacing` : tuple or list (e.g [1.,1.,1.])
            New spacing in mm. If None, must provide `new_size`
        `new_size` : tuple or list of ints (e.g. [100, 100, 100])
            If None, must provide `new_spacing`
        `interp` : string or list/tuple of string
            possible values from this set: {'linear', 'nearest', 'bspline'}
            Different types of interpolation can be provided for each input,
            e.g. for two inputs, `interp=['linear','nearest']

        """
        if new_spacing is None:
            assert new_size is not None, "new_spacing or new_size must be provided"
            assert len(new_size) == 3, "new_size must be of length 3 (x, y, z)"
            self.set_spacing = False
        else:
            assert new_size is None, "cannot provide both new_spacing and new_size"
            assert len(new_spacing) == 3, "new_spacing must be of length 3"
            self.set_spacing = True
        self.new_spacing = new_spacing
        self.new_size = new_size
        self.interp = interp

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            assert isinstance(_input, sitk.SimpleITK.Image), 'input not an image!'
            in_size = _input.GetSize()
            in_spacing = _input.GetSpacing()
            if self.set_spacing:
                out_spacing = self.new_spacing
                out_size = [int(math.ceil(in_size[0]*(in_spacing[0]/out_spacing[0]))),
                            int(math.ceil(in_size[1]*(in_spacing[1]/out_spacing[1]))),
                            int(math.ceil(in_size[2]*(in_spacing[2]/out_spacing[2])))]
            else:
                out_size = self.new_size
                out_spacing = [in_spacing[0]*in_size[0]/out_size[0],
                               in_spacing[1]*in_size[1]/out_size[1],
                               in_spacing[2]*in_size[2]/out_size[2]]
            outputs.append(self._resample(
                _input, out_spacing, out_size, interp[idx]))
        return outputs if idx > 0 else outputs[0]

    def _resample(self, img, out_spacing, out_size, interp):
        if interp == 'linear':
            interp_func = sitk.sitkLinear
        elif interp == 'nearest':
            interp_func = sitk.sitkNearestNeighbor
        elif interp == 'bspline':
            interp_func = sitk.sitkBSpline
        else:
            assert False, "only linear, nearest and bspline interpolation supported"

        resampled_img = sitk.Resample(img, out_size, sitk.Transform(), interp_func,
                                      img.GetOrigin(), out_spacing, img.GetDirection(),
                                      0.0, img.GetPixelIDValue())
        return resampled_img
