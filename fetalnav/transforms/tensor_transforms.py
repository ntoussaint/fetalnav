# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
Relevant Transforms for 3D Biomedical Images.
Mostly adapted from
1. https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
2. https://github.com/ncullen93/torchsample/tree/master/torchsample/transforms

"""
# Author: Bishesh Khanal <bisheshkh@gmail.com>
# 	  King's College London, UK
#	  Department of Computing, Imperial College London, UK

import torch
import numpy as np
import random

def torchwhere(cond, x1, x2):
    """
    Function similar to numpy.where
    Arguments
    ---------
    cond : Tensor or Variable containing Tensor with elements of the same type
    as x1 and x2.
        Checks for this condition element-wise.
    x1 : scalar
        value set to elements
    """
    return (cond * x1) + ((1-cond) * x2)


def torchflip(x, axis):
    """
    Function similar to numpy.flip()
    Taken from here ( dmarnerides' answer): https://github.com/pytorch/pytorch/issues/229
    :TODO: This might not be efficient; update this with torch flip once it is implemented

    Arguments
    ---------
    x : Tensor to be flipped
    axis : axis at which to flip

    Returns
    -------
    out_tensor : Flipped tensor

    """

    dim = x.dim() + axis if axis < 0 else axis
    return x[tuple(slice(None, None) if i != axis
                   else torch.arange(x.size(i)-1, -1, -1).long()
                   for i in range(x.dim()))]

def torchcrop(x, start_idx, crop_sz):
    """
    Arguments
    ---------
    x : Tensor to be cropped
        Either of dim 2 or of 3
    start_idx: tuple/list
        Start indices to crop from in each axis
    crop_sz: tuple/list
        Crop size

    Returns
    -------
    cropped tensor
    """
    dim = len(x.shape) # numpy has .ndim while torch only has dim()!
    assert dim >= 1 and dim <=3, 'supported dimensions: 1, 2 and 3 only'
    if dim == 1:
        return x[start_idx[0]:start_idx[0]+crop_sz[0]]
    elif dim == 2:
        return x[start_idx[0]:start_idx[0]+crop_sz[0],
                 start_idx[1]:start_idx[1]+crop_sz[1]]
    else:
        return x[start_idx[0]:start_idx[0]+crop_sz[0],
                 start_idx[1]:start_idx[1]+crop_sz[1],
                 start_idx[2]:start_idx[2]+crop_sz[2]]


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        """
        Composes (chains) several transforms together into
        a single transform
        Arguments
        ---------
        transforms : a list of transforms
            transforms will be applied sequentially
        """
        self.transforms = transforms

    def __call__(self, *inputs):
        for transform in self.transforms:
            if not isinstance(inputs, (list,tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs


class AddChannel(object):

    def __init__(self, axs=0):
        """Add channel at the chosen input axes of torch tensors
        Useful making input (H x W x D) tensor to (C x H x W x D) tensor
        Arguments
        ---------
        axs: int or sequence of ints:
            Axes at which to add channel, default 0.

        """
        self.axs = axs

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        inputs : Tensors
            Tensors to which channels are added
        Returns
        -------
        outputs: Tensors
        """
        if not isinstance(self.axs, (tuple,list)):
            axs = [self.axs]*len(inputs)
        else:
            axs = self.axs
        outputs = []
        for idx, _input in enumerate(inputs):
            outputs.append(_input.unsqueeze(axs[idx]))
        return outputs if idx > 0 else outputs[0]


class Binarize(object):

    def __init__(self, thresholds = None, lv_uv = (0,1)):
        """
        Binarize input tensor using input threshold.
        Arguments
        ---------
        thresholds : float or sequence of floats
            threshold values. If None, DO NOTHING
        lv_uv : tuple or sequence of tuples
            lower value, upper value such that:
            if val < threshold val = lv else val = uv
            for the output tensor
        """
        self.thresholds = thresholds
        self.lv_uv = lv_uv

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        inputs : Tensors
            Tensors to be binarized
        Returns
        -------
        outputs: Tensors
            Tensors containing just two values: lv and uv
        """

        if not isinstance(self.thresholds, (tuple,list)):
            thresholds = [self.thresholds]*len(inputs)
        else:
            thresholds = self.thresholds
        if not isinstance(self.lv_uv[0], (tuple,list)):
            lv_uv = [self.lv_uv]*len(inputs)
        else:
            lv_uv = self.lv_uv
        outputs = []
        for idx, _input in enumerate(inputs):
            outputs.append(
                torchwhere((_input<thresholds[idx]).float(), *lv_uv[idx]))
        return outputs if idx > 0 else outputs[0]

class Flip(object):

    def __init__(self, axis = 2, flip_prob=0.5):
        """
        Flip with the given flip-noflip probability in the desired axis
        Arguments
        ---------
        axis : int or list of ints
            Flip axis or list of flip axes
        flip_prob : float in the interval [0, 1]
            flip-noflip probability
        """
        self.axis = axis
        assert flip_prob >= 0 and flip_prob <= 1, "flip prob must be between 0 and 1"
        self.flip_prob = flip_prob

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        inputs : Tensors
            Tensors to be flipped
        Returns
        -------
        outputs: Flipped Tensors depending on flip probability.
        Either all tensors are flipped or none are flipped: depends on flip-probability outcome.
        """

        if not isinstance(self.axis, (tuple,list)):
            axes = [self.axis]*len(inputs)
        else:
            axes = self.axis

        flip = False
        if self.flip_prob >= random.random():
            flip=True

        # print(f'flip-prob={self.flip_prob}, flip={flip}, axes={axes}')
        outputs = []
        for idx, _input in enumerate(inputs):
            if flip:
                outputs.append(torchflip(_input, axes[idx]))
            else:
                outputs.append(_input)
        return outputs if idx > 0 else outputs[0]


class PadMultipleOf(object):

    def __init__(self, multiple_of=None, mode='constant', pad_vals=0):
        """
        Pad one or more torch tensors or numpy ndarrays such that:
        The size of the tensors in each dimension is a multiple of `multiple_of`

        Arguments
        ---------
        multiple_of : int or tuple/list of ints
            The number of which the corresponding input tensor's size is to be made
            multiple of. If None, returns the input tensor UNMODIFIED.
        mode: string or tuple/list of strings
            numpy modes in np.pad()
            Note: Not all modes supported because there is no **kwargs here.
        pad_val: int or tuple/list of ints
            Value(s) to be set on padded regions. Only used for mode = 'constant'

        """
        self.m_of = multiple_of
        self.mode = mode
        self.pad_vals = pad_vals

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : Tensor (cpu) or numpy.ndarray
            Tensor/arrays to be padded
        Returns
        -------
            Padded Tensors or ndarrays
        """
        if not isinstance(self.m_of, (tuple,list)):
            self.m_of = [self.m_of]*len(inputs)
        if not isinstance(self.mode, (tuple,list)):
            self.mode = [self.mode]*len(inputs)
        if not isinstance(self.pad_vals, (tuple,list)):
            self.pad_vals = [self.pad_vals]*len(inputs)

        outputs = []
        for idx, _input in enumerate(inputs):
            if self.m_of[idx] is not None:
                _input = self._pad_mof(_input, self.m_of[idx], self.mode[idx], self.pad_vals[idx])
            outputs.append(_input)
        return outputs if idx > 0 else outputs[0]

    def _pad_mof(self, v, m_of, mode, pad_val):
        """
        Arguments
        ---------
        v : Tensor (cpu) or numpy.ndarray
            Tensor/array to be padded
        m_of : self.multiple_of
        mode : self.mode
        Returns
        -------
            Padded Tensor
        """
        is_nparray = True
        if not isinstance(v, np.ndarray):
            v = v.numpy()
            is_nparray = False
        pad_width = []
        for i in range(len(v.shape)):
            pad_width.append(self._symmetric_padwidth(v.shape[i], m_of))

        if mode == 'constant':
            v_ = np.pad(v, pad_width, mode, constant_values=pad_val)
        else:
            v_ = np.pad(v, pad_width, mode)

        if is_nparray:
            return v_
        else:
            return torch.from_numpy(v_)

    def _symmetric_padwidth(self, size, m_of):
        """
        Generate as symmetric as possible pad-width.
        Arguments
        ---------
        size: int
            input size for which the padding width is to be computed
        m_of: int
            pad_width such that the output size will be multiple of this number
        """
        rem =  size % m_of
        if rem == 0:
            pad = (0, 0)
        else:
            pad_len = m_of - rem
            sym = pad_len // 2
            pad = (sym, sym + (pad_len%2))
        return pad



class CropTo(object):

    def __init__(self, crop_size, ref_center=None, label_val=None,
                 label_idx=None, include_ref=True, rand_shift_prob=None):
        """
        Crop a tensor to the given size.
        Arguments
        ---------
        `crop_size`: tuple or list of size 3
            crop size, should be less or equal to input tensor size
        `ref_center`: tuple or list of size 3
            reference position considered as the initial center position to crop the region.
            If None, you may provide label_val that will be used to compute ref_center
        `label_val`: int
            label value from which ref_center is computed using label image.
            If not None, ref_center must be None and label_idx must be provided.
            If both ref_center and label_val are None, ref_center is set to center of the
            input tensor.
        `label_idx` : int
            index that specifies which of the input tensors is to be taken as label tensor
            E.g. if label_idx == 1, positions in inputs[1] tensor where label_val value is present
            is recorded. One of these recorded positions is set to ref_center.
        `include_ref`: boolean
            If True, the cropped region will always include the ref_center position
            (not necessarily in the center if the sizes don't allow, or random shifting!)
        `rand_shift_prob`: float in [0, 1]
            if not None, with this probability randomly shift the cropped region center from
            ref_center
        """
        self.crop_size = crop_size
        self.ref_center = ref_center
        self.label_idx = label_idx
        self.label_val = label_val
        self.include_ref = include_ref
        self.rand_shift_prob = rand_shift_prob
        if self.label_val is not None:
            assert self.ref_center is None, "at least one of ref_center and label_val must be None"
            assert self.label_idx is not None, "label_idx must be provided when using label_val"
        self.crop_start_idx = None # Will be updated when actual cropping is done

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        inputs : Tensors
            tensors to be cropped. If label_val is not None, ref_center is computed from
            inputs[label_idx] tensor.
            i.e. ref_center will be one of the positions where label_val is present in
            inputs[label_idx] tensor.

        Returns
        -------
        cropped outputs for all the inputs including inputs[label_idx]!
        """

        if self.label_idx is not None:
            assert self.label_val is not None, "not None label_idx requires not None label_val!"
            assert self.label_idx < len(inputs), "not enough inputs for given label_idx!"
            # Get all the indices where inputs[label_idx] tensor has value equal to label_val
            label_val_indices = torch.nonzero(inputs[self.label_idx]==self.label_val)
            num_of_indices = label_val_indices.shape[0] # Indices are arranged row-wise
            if num_of_indices > 0: # Randomly select one of these indices
                self.ref_center = tuple(label_val_indices[random.randint(0,num_of_indices-1)])
        #Set up random shifting of self.ref_center
        rand_shift=False
        if self.rand_shift_prob is not None:
            if self.rand_shift_prob >= random.random():
                rand_shift = True
        #print(f'ref_center for cropping = {self.ref_center}')
        # return self._crop_to(self.crop_size, ref_center=self.ref_center,
        #                      include_ref=self.include_ref, random_shift=rand_shift,
        #                      *inputs)
        # NEED TO UNDERSTAND python 3 positional vs named args. Does not work when providing
        # all as named args, and then providing *args.
        return self._crop_to(self.crop_size, self.ref_center, self.include_ref,
                             rand_shift, *inputs)

    def pairwise_exclusive(self, iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    def _crop_to(self, crop_size, ref_center=None, include_ref=True, random_shift=False, *imgs):
        img_sizes = [list(img.shape) for img in imgs]
        if len(img_sizes) > 1:
            assert img_sizes[1:] == img_sizes[:-1], 'all images of be of same size'
        img_size = img_sizes[0]

        # If tensor size equals crop size, return whole tensors
        if(all([i_s == c_s for i_s, c_s in zip(img_size, crop_size)])):
            return imgs if len(imgs) > 1 else imgs[0]
        assert(all([i_s >= c_s for i_s, c_s in
                    zip(img_size, crop_size)])), 'crop size must be <= input tensor'
        img_center = [x//2 for x in img_size]
        if ref_center is None:
            ref_center = img_center
        #print(f'ref_center={ref_center}')
        crop_start_idx = [x-y//2 for x, y in zip(ref_center, crop_size)]
        crop_end_idx = [x+y for x,y in zip(crop_start_idx, crop_size)]
        # Displace start_idx to zero if it is negative
        crop_start_idx = [0 if x<0 else x for x in crop_start_idx]
        # Displace start_idx such that end_idx is within img
        outside_offset = [(i_sz-end_idx) for i_sz,end_idx in
                          zip(img_size, crop_end_idx)]
        crop_start_idx = [x+y if y<0 else x for x,y in zip(crop_start_idx, outside_offset)]
        crop_end_idx = [x+y for x,y in zip(crop_start_idx, crop_size)]
        #print(f'before random shift: crop_start_idx={crop_start_idx}')
        if random_shift:
            # i_sz is idx+1, same way c_end is excluded idx so no extra -1 here.
            offset_range = [(-c_st, i_sz-c_end) for c_st,i_sz,c_end in
                                zip(crop_start_idx, img_size, crop_end_idx)]
            if include_ref: # limit offset range to include ref_center
                # Note that c_end has value of index that is not included!
                # Hence extra -1 when using this as index with other normal indices!
                offset_range = [(-min(-x[0], c_end-1-r_center), min(x[1], r_center-c_st))
                                for x,c_st,c_end,r_center in
                                zip(offset_range, crop_start_idx, crop_end_idx, ref_center)
                               ]
            offset = [random.randint(*x) for x in offset_range]
            crop_start_idx = [x+y for x,y in zip(crop_start_idx, offset)]
            crop_center = [x+y//2 for x,y in zip(crop_start_idx, crop_size)]
            #print(f'after random shift: crop_start_idx={crop_start_idx}')
        cropped_imgs = []
        for idx, img in enumerate(imgs):
            cropped_imgs.append(torchcrop(img, crop_start_idx, crop_size))
        self.crop_start_idx = crop_start_idx
        return cropped_imgs if idx > 0 else cropped_imgs[0]

    def get_refcenter(self):
        return self.ref_center

    def get_crop_start_idx(self):
        return self.crop_start_idx


class ResizeCropOrPad(object):

    def __init__(self, new_size, pad_vals=0):
        """
        Resize one or more torch tensors or numpy ndarrays to the given (new) size.
        In any given dimension, either crop or pad depending on whether the new size
        is less or greater than the original size.

        Arguments
        ---------
        new_size : tuple/list of ints
            New size in each dimensions.
        pad_vals : int or tuple/list of ints
            Value(s) to be set on padded regions

        """
        self.new_size = new_size
        self.pad_vals = pad_vals

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : Tensor (cpu) or numpy.ndarray
            Tensor/arrays to be resized
        Returns
        -------
            Resized Tensors or ndarrays
        """
        if not isinstance(self.pad_vals, (tuple,list)):
            self.pad_vals = [self.pad_vals]*len(inputs)

        outputs = []
        for idx, _input in enumerate(inputs):
            _input = self._crop_or_pad(_input, self.new_size, self.pad_vals[idx])
            outputs.append(_input)
        return outputs if idx > 0 else outputs[0]

    def _crop_or_pad(self, x, new_size, pad_val):
        """
        Arguments
        ---------
        x : Tensor (cpu) or numpy.ndarray
            Tensor/array to be padded
        new_size : new size
        Returns
        -------
            Padded/Cropped Tensor or numpy.ndarray
        """
        is_nparray = True
        # Convert to numpy array, this is required for padding!
        # If just cropping, no need to convert to numpy array

        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy()
            is_nparray = False
        for ax, ax_size in enumerate(new_size):
            if ax_size < x.shape[ax]:
                crop_size = list(x.shape)
                crop_size[ax] = ax_size
                x = CropTo(crop_size)(x)
            elif ax_size > x.shape[ax]:
                pad_widths = np.zeros((len(x.shape), 2), dtype='uint').tolist()
                extra = ax_size - x.shape[ax]
                sym = extra // 2
                pad_widths[ax] = [sym, sym + (extra % 2)]
                x = np.pad(x, pad_widths, 'constant', constant_values=pad_val)
        if is_nparray:
            return x
        else:
            return torch.from_numpy(x)





class PaddToRatio(object):
    """Pad a tensor or nd array to obtain a certain aspect ratio.
    Padding an image in x or y direction in order to obtain
    a specific aspect ratio given as argument

    The input should be a tensor or nd array arranged as ****xdimyxdimx
    """

    def __init__(self, outputaspect):
        self.outputaspect = outputaspect

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : numpy ndarrays
            Images to be padded.
        Returns
        -------
        Numpy nd arrays padded to a certain ratio
        """
        outputs = []

        for idx, _input in enumerate(inputs):
            outputs.append(self._paddtoratio(_input,self.outputaspect))
        return outputs if idx > 0 else outputs[0]


    def _paddtoratio(self, im, outputaspect):
        size = im.shape
        y_id = len(size) - 2
        x_id = len(size) - 1
        x_dimension = size[x_id]
        y_dimension = size[y_id]
        inputaspect = y_dimension/float(x_dimension)
        newsize = list(size)
        if inputaspect > outputaspect:
            newsize[x_id] += int(( y_dimension / outputaspect - x_dimension ))
        else:
            newsize[y_id] += int(( outputaspect * x_dimension - y_dimension ))
        padder = ResizeCropOrPad(newsize)
        return padder(im)


class CropToRatio(object):
    """Crop a tensor or nd array to obtain a certain aspect ratio.
    Padding an image in x or y direction in order to obtain
    a specific aspect ratio given as argument

    The input should be a tensor or nd array arranged as nBatchesXnChannelsXxXy
    """

    def __init__(self, outputaspect):
        self.outputaspect = outputaspect

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : numpy ndarrays
            Images to be padded.
        Returns
        -------
        Numpy nd arrays padded to a certain ratio
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            outputs.append(self._croptoratio(_input,self.outputaspect))
        return outputs if idx > 0 else outputs[0]


    def _croptoratio(self, im, outputaspect):
        size = im.shape
        y_id = len(size) - 2
        x_id = len(size) - 1
        x_dimension = size[x_id]
        y_dimension = size[y_id]
        inputaspect = y_dimension/float(x_dimension)
        newsize = list(size)
        if inputaspect > outputaspect:
            newsize[y_id] = int(size[y_id] * outputaspect / inputaspect)
        else:
            newsize[x_id] = int(size[x_id] * inputaspect / outputaspect)
        cropper = ResizeCropOrPad(newsize)
        return cropper(im)




class Resize(object):
    """Resize a 2D image using scipy.misc.imresize(), only works in 2D as
    it is using the imresize() method from numpy
    The input needs to be of type b x c x w x h
    The last 2 dimensions are taken fo bilinear resampling
    """

    def __init__(self, size, interp='bilinear'):
        self.size = size
        self.interp = interp

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : tensors or numpy ndarrays
            Images to be resized, of dimensions 3 or 4: c x w x h or b x c x w x h.
        Returns
        -------
        torch tensors resized to a certain size
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            outputs.append(self._resize(_input,self.size, self.interp))
        return outputs if idx > 0 else outputs[0]


    def _resize(self, im, size, interp='bilinear'):
        from scipy.misc import imresize

        assert len(size) >= 2, "size dimension smaller than 2 is not supported"
        assert (len(im.shape) == 3) or (len(im.shape) == 4), "Input tensor/array shape not supported"

        is_nparray = True
        # Convert to numpy array, this is required for padding!
        # If just cropping, no need to convert to numpy array
        if not isinstance(im, np.ndarray):
            im = im.cpu().numpy()
            is_nparray = False

        newsize = list(im.shape)
        newsize[len(newsize)-2] = size[len(size)-2]
        newsize[len(newsize)-1] = size[len(size)-1]

        ret = np.ndarray(newsize, dtype=im.dtype)

        for i in range(newsize[0]):
            if len(im.shape) == 3:
                s = np.squeeze(im[i,:,:])
                ret[i,:,:] = imresize(s, size, interp)
            elif len(im.shape) == 4:
                for j in range(newsize[1]):
                    s = np.squeeze(im[i,j,:,:])
                    ret[i,j,:,:] = imresize(s, size, interp)
        if is_nparray:
            return ret
        else:
            return torch.from_numpy(ret)




class Slices2D(object):

    def __init__(self, view_dir=0, slice_num=None):
        """
        Extract slices from a tensor with size nBatchesXnChannelsXhXwXd.
        The extracted slices reduces the tensor to nBatchesXnChannelsXuXv
        where u, v are the dimension of the extracted slices.
        Th size of u, v depends on the view_dir to decide on of three axes: hXw, hXd or wXd

        Arguments
        ---------
        view_dir : int {0, 1, 2}
            Return slice of 3d image from this direction
        slice_num : int (default: None)
            Return this slice from given 3d array and view_dir
            If None, returns a middle slice of the given view direction.
        """

        self.view_dir = view_dir
        self.slice_num = slice_num

    def __call__(self, tensor):
        """
        Arguments
        ---------
        tensor : Tensor
            Tensor from which slices are to be extracted
        Returns:
        --------
        list of tensors: List of 2d tensors
        """

        assert self.view_dir in [0, 1, 2], 'invalid view direction'
        vol_size = tensor.shape[2:]
        slice_num = self.slice_num
        if slice_num is None:
            slice_num = vol_size[self.view_dir] // 2
        assert slice_num < vol_size[self.view_dir], 'requested slice not available in input array '
        if self.view_dir == 0:
            return tensor[:,:,slice_num,:,:]
        elif self.view_dir == 1:
            return tensor[:,:,:,slice_num,:]
        else:
            return tensor[:,:,:,:,slice_num]


class Standardize(object):
    """
    Standardize input tensor: tensor_out = (tensor - mean) / std
    :TODO:
    1. Add batch_norm boolean argument to choose computing mean and std of each sample
       in the mini-batch independently or together.
    """

    def __init__(self, mean_std=None, #TODO: batch_norm = False
                 mask_val=None, mask_eps=1e-6, mask_replace_val=None):
        """
        Standardize input tensor, i.e. change to Z-score values.

        Arguments
        ---------
        mean_std : tuple/list of floats of size two
            Mean and standard deviation used for computing z-score values.
            If None, computes them from the input tensor either for the whole mini-batch
            or independently for each sample depending on batch_norm input.
        mask_val: float
            Value which is not used in computing or setting z-score values.
            Any x in input tensor with abs(mask_val-x) < mask_eps will not be used
            in computing z-score. Also, these values will be replaced with mask_replace_val.
        mask_eps: float
            See mask_val description.
        mask_replace_val: type of elements of input tensor
            See mask_val description. If None, all the positions in input tensor having
            values mask_val are not modified.
        (TODO: NOT IMPLEMENTED YET)batch_norm : Boolean
            If mean_std in not None, this input is irrelevant.
            The first dimension of the input tensor is assumed to be of mini-batch.
            If False, z-score is computed independently for each input sample of this mini-batch.
        """
        #self.batch_norm = batch_norm
        self.mean_std = mean_std
        self.mask_val = mask_val
        self.mask_eps = mask_eps
        self.mask_replace_val = mask_replace_val


    def __call__(self, tensor):
        """
        Arguments
        ---------
        tensor : Tensor
            Tensor of size (B, tensor-dimensions)

        Returns:
        --------
            Tensor: Standardized tensor
        """
        # Get mean and standard deviation
        if self.mean_std:
            mn, std = self.mean_std[0], self.mean_std[1]
        else:
            if self.mask_val is None:
                mn, std = tensor.mean(), tensor.std()
            else:
                mask = torch.abs(tensor-self.mask_val) > self.mask_eps
                arr = torch.masked_select(tensor, mask)
                mn, std = arr.mean(), arr.std()

        output = (tensor - mn)/std

        # Replace masked out regions with desired values
        if self.mask_val is not None:
            if self.mask_replace_val is not None:
                output[~mask] = self.mask_replace_val
            else:
                output[~mask] = self.mask_val

        return output


class Rescale(object):

    def __init__(self, interval=(0, 1)):
        """
        Rescales input tensor to [a, b]
           tensor_out = (x - min(x)) / (max(x) - min(x))
        Arguments
        ---------
        interval : tuple   (default: [0, 1])
            [min, max]
        :TODO: Add support for rescaling multiple tensors to make it consistent with
        other transforms.
        """

        import collections
        assert (isinstance(interval, collections.Iterable) and len(interval) == 2)
        self.interval = interval

    def __call__(self, tensor):
        """
        Arguments
        ---------
        tensor : Tensor
            Tensor of size (C, H, W, D)

        Returns:
        --------
        Tensor: in the range [a, b]
        https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
        """
        a, b = self.interval[0], self.interval[1]
        eps = 1e-5
        den = tensor.max() - tensor.min()
        if (abs(den)) < eps: # If the difference is zero, take max
            den = tensor.max()
        if (abs(den) < eps): # If the max is also zero (implies everything zero), return a.
            tensor.fill_(a)
            return tensor
        else:
            return ((
                (b-a)*(tensor - tensor.min())/den
                ) + a)


class OneHot(object):
    """
    Transform input tensor to a one-hot encoding along the channel dimension.
    """

    def __init__(self, n_labels):
        """

        Arguments
        ---------
        n_labels: int or tuple/list of ints
            number of labels to create one-hot encoding for
            The output nChannels will equal to n_labels.
        """
        self.n_labels = n_labels

    def __call__(self, *inputs):
        """
        Transform input tensor to a one-hot encoding along the channel dimension.
        Arguments
        ---------
        inputs : LongTensor or ByteTensor
            These will be converted to LongTensors before being used.

        Returns
        -------
        one hot encoded outputs
        """
        if not isinstance(self.n_labels, (tuple,list)):
            self.n_labels = [self.n_labels]*len(inputs)

        outputs = []
        for idx, _input in enumerate(inputs):
            in_size = tuple(_input.shape)
            out = torch.LongTensor(
                in_size[0], self.n_labels[idx], *in_size[2:])
            out.zero_()
            out.scatter_(1, _input, 1)
            # along dim 1 (1st arg), set 1 (3rd arg) at index given by _input (2nd arg)
            # The values along dim 1 of _input are the indices where 1's are to be set
            # in out along this same dim 1
            outputs.append(out)
        return outputs if idx > 0 else outputs[0]


class NormalizeWithThreshold(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, threshold):
        self.mean = mean
        self.std = std
        self.threshold = threshold

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return self._normalizewiththreshold(tensor, self.mean, self.std, self.threshold)



    def _normalizewiththreshold(self, tensor, mean, std, threshold):
        """Normalize a tensor image to a certain mean and standard deviation taking only pixels
        above a certain threshold.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.
            threshold (sequence): Sequence of lower-threshold to use for each channel
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not _is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
        # TODO: make efficient
        for t, m, s, thr in zip(tensor, mean, std, threshold):
            t_idx = (t > thr)
            mt=t[t_idx].mean()
            st=t[t_idx].std()
            t[t_idx] = t[t_idx].sub(mt).div(st)
            t[t_idx] = t[t_idx].mul(s).sub(-m)

        return tensor
