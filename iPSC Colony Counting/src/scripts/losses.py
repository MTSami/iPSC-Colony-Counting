__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import math
import torch
import sklearn
from sklearn.utils.extmath import cartesian
import numpy as np
from torch.nn import functional as F
import os
import time
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KernelDensity
import skimage.io
from matplotlib import pyplot as plt
from torch import nn


torch.set_default_dtype(torch.float32)


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances


def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    between two unordered sets of points (the function is symmetric).
    Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res


###############################################################################################
#########  Sami Center FOCAL LOSS #############################################################
###############################################################################################

def focal_loss(pred, gt, gamma = 2, reduction = 'mean'):
    
    ##########################Debug#### 
    # print(type(gt))
    # print(type(pred))
    # print("GT size: ", gt.size())
    # print("GT dtype", gt.dtype)
    # print("Pred size: ", pred.size())
    # print("Pred Max: ", torch.max(pred))
    # print("Pred Min: ", torch.min(pred))
    ##########################Debug#### 
    
    # Gather probability where y=1.0 and 1-probability otherwise
    p_t = torch.where(gt == 1.0, pred, 1.0 - pred)
    # Apply modulating term
    loss = -torch.pow(1.0 -p_t, gamma) * F.logsigmoid(p_t)
    
    # Reduce loss tensor
    if reduction == 'sum':
      loss = torch.sum(loss)
    else:
      loss = torch.mean(loss)
      
    return loss


def center_focal_loss(pred, gt, alpha = 2, beta = 4):
    
    '''
    Arguments:
        pred (h x w)
        gt_regr (h x w)
    '''
    
    # Debug#### 
    # with torch.no_grad():
    #     print(type(gt))
    #     print(type(pred))
    #     print("GT size: ", gt.size())
    #     print("GT dtype", gt.dtype)
    #     print("Pred size: ", pred.size())
    #     print("Pred Max: ", torch.max(pred))
    #     print("Pred Min: ", torch.min(pred))
    # Debug####


    ## Create zeros and ones masks
    zeros = torch.zeros_like(gt).float()
    ones = torch.ones_like(gt).float()
    
    # Create poistive prediciton mask and negative prediciton mask
    pos_pred_mask = torch.where(gt == 1.0, ones, zeros).float()
    neg_pred_mask = torch.where(gt < 1.0, ones, zeros).float()

    # Postive Match Term
    pos_term = torch.pow(1-pred, alpha) * torch.log(pred) * pos_pred_mask
    
    # Negative Match Term
    neg_term = torch.pow(1-gt, beta) * torch.pow(pred, alpha) * torch.log(1-pred) * neg_pred_mask
    
    # Normalization factor
    pos_count  = pos_pred_mask.sum()
    # print("pos_count: ", pos_count)
    
    #Add both terms and normalize
    loss = (pos_term + neg_term).float().sum()
    loss = -(loss / pos_count)
    
    return loss


class FocalLoss(nn.Module):
    
    def __init__(self, resized_height, resized_width, device = torch.device('cpu')):
        
        super(nn.Module, self).__init__()
        
        ## Currently needed for mahd metric, will remove later
        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
    
    def forward(self, center_prob_maps, center_gt_maps):
        
        ##############################################################
        # Convert center_gt_maps and from list to tensor
        ##############################################################
        # with torch.no_grad():
        #     print("center_prob_maps: ", center_prob_maps.size())
        #     print("center_gt_maps: ", center_gt_maps.size())
            
            # test_center_gt_maps = center_gt_maps[0]
            # print("test_center_gt_maps shape:", test_center_gt_maps.size())
            # test_center_gt_count = test_center_gt_maps.eq(1).sum()
            # print("test_center_gt_count: ", test_center_gt_count)
            # print("test_center_gt_max: ", torch.max(test_center_gt_maps))
            # print("test_center_gt_min: ", torch.min(test_center_gt_maps))
        
        ##############################################################
        
        terms_1 = []

        batch_size = center_prob_maps.shape[0]
        BCE_loss = nn.BCELoss(reduction='mean')
        
        ## Loop each item in batch
        for b in range(batch_size):
            
            center_prob_maps_b = center_prob_maps[b, :, :]
            center_gt_maps_b = center_gt_maps[b, :, :] #Sami: Pick the gt_map for the batch index b
            ## Get focal loss
            f_loss = center_focal_loss(center_prob_maps_b, center_gt_maps_b)
            terms_1.append(f_loss)

        terms_1 = torch.stack(terms_1)
        
        res = terms_1.mean()
        
        return res
###############################################################################################
###############################################################################################
###############################################################################################




class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, resized_width,
                 p=-9,
                 return_2_terms=False,
                 device=torch.device('cpu')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(nn.Module, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())

        self.return_2_terms = return_2_terms
        self.p = p
    
    

    ##################SAMI EDITS############################################
    def forward(self, prob_map, gt_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt_map: (B x H x W) List of the probability map generated from gt mask.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :param orig_widths: List of the original widths for each image
                            in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """
        
        ### Convert every row to torch tensor 
        list_of_tensors = [torch.tensor(data) for data in gt_map]
        ### Stack torch tensors 
        gt_map = torch.stack(list_of_tensors)
        gt_map = torch.squeeze(gt_map, 1)
        
        # print("prob_map: ", prob_map.size())
        # print("gt_map: ", gt_map.size())

        _assert_no_grad(gt)
        _assert_no_grad(gt_map) #Sami

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        
        assert batch_size == len(gt)
        assert batch_size == gt_map.shape[0]

        terms_1 = []
        terms_2 = []
        
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_map_b = gt_map[b, :, :] #Sami: Pick the gt_map for the batch index b
            # print("prob_map_b: ", prob_map_b.size())
            # print("gt_map_b: ", gt_map_b.size())
            
            ################################Center Focal loss###############################
            center_f_loss = center_focal_loss(prob_map_b, gt_map_b)
            terms_3 = center_f_loss
            # print("Center Focal LOSS:", terms_3)
            ################################################################################
            


            ################################Hausdorff Loss###############################
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b/self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) *\
                self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1)*gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated)*self.max_dist + p_replicated*d_matrix
            minn = generaliz_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        
        
        with torch.no_grad():
            if self.return_2_terms:
                res = terms_1.mean() + terms_2.mean(), terms_3
            else:
                res = terms_1.mean() + terms_2.mean() + terms_3

        return res
        ########################################################################


def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    # print()
    # print("Input Shape: ", tensor.size())
    assert p < 0
    # res = torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    res = torch.min(tensor, dim = 0)[0]
    # res = torch.min(tensor)
    # print("Output Shape: ", res.shape)
    # print()
    
    return res


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
