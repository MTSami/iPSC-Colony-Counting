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

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn.metrics
import sklearn.neighbors
import scipy.stats
from . import losses

class Judge():
    """
    A Judge computes the following metrics:
        (Location metrics)
        - Precision
        - Recall
        - Fscore
        - Mean Average Hausdorff Distance (MAHD)
        (Count metrics)
        - Mean Error (ME)
        - Mean Absolute Error (MAE)
        - Mean Percent Error (MPE)
        - Mean Absolute Percent Error (MAPE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Pearson correlation (r)
        - Coefficient of determination (R^2)
        (Segmentation metrics)
        - Mean Pixel Accuracy (MPA)
        - Mean Intersection over Union (MIOU)
        - Mean Dice Coefficient (MDC)
    """

    def __init__(self, r):
        """
        Create a Judge that will compute metrics with a particular r
         (r is only used to compute Precision, Recall, and Fscore).

        :param r: If an estimated point and a ground truth point 
                  are at a distance <= r, then a True Positive is counted.
        """
        # Location metrics
        self.r = r   ## radius around each gt center
        self.tp = 0  ##(True Positive)
        self.fp = 0  ##(False Positive)
        self.fn = 0  ##(False Negative)

        # Count data points
        self._predicted_counts = []
        self._true_counts = []

        # Internal variables
        self._sum_ahd = 0  ##(Average Hasdorff Distance)
        self._sum_e = 0    ##(Error)
        self._sum_pe = 0   ##(Percent Error)
        self._sum_ae = 0   ##(Absolute Error)
        self._sum_se = 0   ##(Squared Error)
        self._sum_ape = 0  ##(Absolute Percent Error)
        self._n_calls_to_feed_points = 0
        self._n_calls_to_feed_count = 0
        
        ###Sami edits
        self._sum_pa = 0  ##(Sum of pixel accuracy)
        self._sum_iou = 0  ##(Sum of IoU)
        self._sum_dc = 0  ##(Sum of Dice coefficient)
        self._n_calls_to_feed_mask = 0
    
    
    def get_neighbor(self, start, dest):
        count = 0
        
        ## For each point in start find dest points
        for st_x, st_y in start:
            for des_x, des_y in dest:
                euclidean_dist = math.sqrt((des_x - st_x) ** 2 + (des_y - st_y) ** 2)
                ##Found a prediction in gt radius
                if euclidean_dist < self.r:
                    count+=1
                    break
                
        
        return count
        

    def feed_points(self, pts, gt, max_ahd=np.inf):
        """
        Evaluate the location metrics of one set of estimations.
         This set can correspond to the estimated points and
         the groundtruthed points of one image.
         The TP, FP, FN, Precision, Recall, Fscore, and AHD will be
         accumulated into this Judge.

        :param pts: List of estmated points.
        :param gt: List of ground truth points.
        :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
        """

        if len(pts) == 0:
            tp = 0
            fp = 0
            fn = len(gt)
        else:
            # nbr = sklearn.neighbors.NearestNeighbors(n_neighbors=1, metric='euclidean').fit(gt)
            # dis, idx = nbr.kneighbors(pts)
            # detected_pts = (dis[:, 0] <= self.r).astype(np.uint8)

            # nbr = sklearn.neighbors.NearestNeighbors(n_neighbors=1, metric='euclidean').fit(pts)
            # dis, idx = nbr.kneighbors(gt)
            # detected_gt = (dis[:, 0] <= self.r).astype(np.uint8)

            # tp = np.sum(detected_pts)
            # fp = len(pts) - tp
            # fn = len(gt) - np.sum(detected_gt)
            
            
            #########SAMI EDITS################################
            ## For GT, every missing Pred
            tp = self.get_neighbor(gt, pts)
            if tp < 0: 
                tp = 0
            
            assert tp <= len(gt), "Can't be more than GT len"
            
            
            fp = len(pts) - tp
            
            if fp < 0: 
                fp = 0
            
            assert fp >= 0, "FP must me more than 1"
            
            ## For Pred, every missing GT
            detected_gt = self.get_neighbor(pts, gt)
            
            fn = len(gt) - detected_gt
            
            if fn < 0: 
                fn = 0
            ########################################################
            

        self.tp += tp
        self.fp += fp
        self.fn += fn

        # Evaluation using the Averaged Hausdorff Distance
        ahd = losses.averaged_hausdorff_distance(pts, gt,
                                                 max_ahd=max_ahd)
        self._sum_ahd += ahd
        self._n_calls_to_feed_points += 1

    def feed_count(self, estim_count, gt_count):
        """
        Evaluate count metrics for a count estimation.
         This count can correspond to the estimated and groundtruthed count
         of one image. The ME, MAE, MPE, MAPE, MSE, and RMSE will be updated
         accordignly.

        :param estim_count: (positive number) Estimated count.
        :param gt_count: (positive number) Groundtruthed count.
        """

        if estim_count < 0:
            raise ValueError(f'estim_count < 0, got {estim_count}')
        if gt_count < 0:
            raise ValueError(f'gt_count < 0, got {gt_count}')

        self._predicted_counts.append(estim_count)
        self._true_counts.append(gt_count)

        e = estim_count - gt_count
        ae = abs(e)
        if gt_count == 0:
            ape = 100*ae
            pe = 100*e
        else:
            ape = 100 * ae / gt_count
            pe = 100 * e / gt_count
        se = e**2

        self._sum_e += e
        self._sum_pe += pe
        self._sum_ae += ae
        self._sum_se += se
        self._sum_ape += ape

        self._n_calls_to_feed_count += 1
    
    ###Sami edits
    def feed_mask(self, pred_mask, gt_mask):
        """
        Evaluate segmentation metrics for a mask estimation.
         This mask can correspond to the estimated and gt mask
         of one image. The MPA, MIOU, and MDC will be updated
         accordignly.

        :param pred_mask: (binary mask) Preducted Segmentation Mask.
        :param gt_count: (binary mask) Ground Truth Segmentation Mask.
        """
        
        self.pred_mask = pred_mask
        self.gt_mask = gt_mask
        
        #### The predicted mask is in the range [0, 1], threshold by (0.5) to get binary {0,1}
        pred_mask_bianry = np.where(self.pred_mask >= 0.5, 1, 0).astype("uint8")
        

        ######### Calculate pixel accuracy ##############################
        pixel_accuracy = 0
        match_idx = np.where(pred_mask_bianry == self.gt_mask)
        matches = len(match_idx[0])
        total_pixels = self.pred_mask.shape[0] * self.pred_mask.shape[1]
        pixel_accuracy = float((matches/total_pixels)*100.0)
        # print(pixel_accuracy)
        #################################################################
        
        ######### Calculate IoU: Foreground ##############################
        iou = 0
        
        intersection = np.sum(pred_mask_bianry * self.gt_mask)
        # print("intersection", intersection)
        union = np.sum(pred_mask_bianry) + np.sum(self.gt_mask) - intersection
        # print("union", union)
        iou = intersection/union
        # print("iou", iou)
        #################################################################
        
        ## Calculate Dice Coefficient: Foreground #######################
        dice_coeff = 0
        
        union = np.sum(pred_mask_bianry) + np.sum(self.gt_mask)
        dice_coeff = (2*intersection)/union
        # print("dice_coeff", dice_coeff)
        
        self._sum_pa += pixel_accuracy
        self._sum_iou += iou
        self._sum_dc += dice_coeff
        #################################################################
        
        self._n_calls_to_feed_mask += 1
        

    @property
    def me(self):
        """ Mean Error (float) """
        return float(self._sum_e / self._n_calls_to_feed_count)

    @property
    def mae(self):
        """ Mean Absolute Error (positive float) """
        return float(self._sum_ae / self._n_calls_to_feed_count)

    @property
    def mpe(self):
        """ Mean Percent Error (float) """
        return float(self._sum_pe / self._n_calls_to_feed_count)

    @property
    def mape(self):
        """ Mean Absolute Percent Error (positive float) """
        return float(self._sum_ape / self._n_calls_to_feed_count)

    @property
    def mse(self):
        """ Mean Squared Error (positive float)"""
        return float(self._sum_se / self._n_calls_to_feed_count)

    @property
    def rmse(self):
        """ Root Mean Squared Error (positive float)"""
        return float(math.sqrt(self.mse))

    @property
    def coeff_of_determination(self):
        """ Coefficient of Determination (-inf, 1]"""
        return sklearn.metrics.r2_score(self._true_counts,
                                        self._predicted_counts)

    @property
    def pearson_corr(self):
        """ Pearson coefficient of Correlation [-1, 1]"""
        return scipy.stats.pearsonr(self._true_counts,
                                    self._predicted_counts)[0]

    @property
    def mahd(self):
        """ Mean Average Hausdorff Distance (positive float)"""
        return float(self._sum_ahd / self._n_calls_to_feed_points)

    @property
    def precision(self):
        """ Precision (positive float) """
        return float(100*self.tp / (self.tp + self.fp)) \
            if self.tp > 0 else 0

    @property
    def recall(self):
        """ Recall (positive float) """
        return float(100*self.tp / (self.tp + self.fn)) \
            if self.tp > 0 else 0

    @property
    def fscore(self):
        """ F-score (positive float) """
        return float(2 * (self.precision*self.recall /
                          (self.precision+self.recall))) \
            if self.tp > 0 else 0
    
    @property
    def mpa(self):
        """
        What percent of predicted pixel belongs to GT
        """
        return float(self._sum_pa / self._n_calls_to_feed_mask)
    
    @property
    def miou(self):
        """
        Intersection over union between predicted mask and gt mask
        """
        return float(self._sum_iou / self._n_calls_to_feed_mask)
    
    @property
    def mdc(self):
        """
        Dice Coefficient between predicted mask and gt mask
        """
        return float(self._sum_dc / self._n_calls_to_feed_mask)


def make_metric_plots(csv_path, taus, radii, title=''):
    """
    Create a bunch of plots from the metrics contained in a CSV file.

    :param csv_path: Path to a CSV file containing metrics.
    :param taus: Detection thresholds tau.
                 For each of these taus, a precision(r) and recall(r) will be created.
                 The closest to each of these values will be used.
    :param radii: List of values, each with different colors in the scatter plot.
                  Maximum distance to consider a True Positive.
                  The closest to each of these values will be used.
    :param title: (optional) Title of the plot in the figure.
    :return: Dictionary with matplotlib figures.
    """

    dic = {}

    # Data extraction
    df = pd.read_csv(csv_path)

    plt.ioff()

    # ==== Precision and Recall as a function of R, fixing t ====
    for tau in taus:
        # Find closest threshold
        tau_selected = df.th.values[np.argmin(np.abs(df.th.values - tau))]
        print(f'Making Precision(r) and Recall(r) using tau={tau_selected}')

        # Use only a particular r
        precision = df.precision.values[df.th.values == tau_selected]
        recall = df.recall.values[df.th.values == tau_selected]
        r = df.r.values[df.th.values == tau_selected]

        # Create the figure for "Crowd" Dataset
        fig, ax = plt.subplots()
        precision = ax.plot(r, precision, 'r--',label='Precision')
        recall = ax.plot(r, recall,  'b:',label='Recall')
        ax.legend()
        ax.set_ylabel('%')
        ax.set_xlabel(r'$r$ (in pixels)')
        ax.grid(True)
        plt.title(title + f' tau={round(tau_selected, 4)}')

        # Hide grid lines below the plot
        ax.set_axisbelow(True)

        # Add figure to dictionary
        dic[f'precision_and_recall_vs_r,_tau={round(tau_selected, 4)}'] = fig
        plt.close(fig)

    # ==== Precision vs Recall ====
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if len(radii) > len(colors):
        print(f'W: {len(radii)} are too many radii to plot, '
              f'taking {len(colors)} randomly.')
        radii = list(radii)
        np.random.shuffle(radii)
        radii = radii[:len(colors)]
        radii = sorted(radii)

    # Create figure
    fig, ax = plt.subplots()
    plt.ioff()
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.grid(True)
    plt.title(title)

    for r, c in zip(radii, colors):
        # Find closest R
        r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

        # Use only a particular r for all fixed thresholds
        selection = (df.r.values == r_selected) & (df.th.values >= 0)
        if selection.any():
            precision = df.precision.values[selection]
            recall = df.recall.values[selection]

            # Sort by ascending recall
            idxs = np.argsort(recall)
            recall = recall[idxs]
            precision = precision[idxs]

            # Plot precision vs. recall for this r
            ax.scatter(recall, precision,
                       c=c, s=2, label=f'$r={r}$')

        # Otsu threshold (tau = -1)
        selection = (df.r.values == r_selected) & (df.th.values == -1)
        if selection.any():
            precision = df.precision.values[selection]
            recall = df.recall.values[selection]
            ax.scatter(recall, precision,
                       c=c, s=8, marker='+', label=f'$r={r}$, Otsu')

        # BMM threshold (tau = -2)
        selection = (df.r.values == r_selected) & (df.th.values == -2)
        if selection.any():
            precision = df.precision.values[selection]
            recall = df.recall.values[selection]
            ax.scatter(recall, precision,
                       c=c, s=8, marker='s', label=f'$r={r}$, BMM')

    # Invert legend order
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    
    # Put legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.03))

    # Hide grid lines below the plot
    ax.set_axisbelow(True)

    # Add figure to dictionary
    dic['precision_vs_recall'] = fig
    plt.close(fig)


    # ==== Precision as a function of tau for all provided R ====
    # Create figure
    fig, ax = plt.subplots()
    plt.ioff()
    ax.set_ylabel('Precision')
    ax.set_xlabel(r'$\tau$')
    ax.grid(True)
    plt.title(title)

    list_of_precisions = []

    for r, c in zip(radii, colors):
        # Find closest R
        r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

        # Use only a particular r for all fixed thresholds
        selection = (df.r.values == r_selected) & (df.th.values >= 0)
        if selection.any():
            precision = df.precision.values[selection]
            list_of_precisions.append(precision)
            taus = df.th.values[selection]

            # Plot precision vs tau for this r
            ax.scatter(taus, precision, c=c, s=2, label=f'$r={r}$')

        # Otsu threshold (tau = -1)
        selection = (df.r.values == r_selected) & (df.th.values == -1)
        if selection.any():
            precision = df.precision.values[selection]
            ax.axhline(y=precision,
                       linestyle='-',
                       c=c, label=f'$r={r}$, Otsu')

        # BMM threshold (tau = -1)
        selection = (df.r.values == r_selected) & (df.th.values == -2)
        if selection.any():
            precision = df.precision.values[selection]
            ax.axhline(y=precision,
                       linestyle='--',
                       c=c, label=f'$r={r}$, BMM')

    if len(list_of_precisions) > 0:
        # Plot average precision for all r's
        ax.scatter(taus, np.average(np.stack(list_of_precisions), axis=0),
                   c='k', marker='x', s=7, label='avg along r')

    

    # Invert legend order
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]

    # Put legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.03))

    # Hide grid lines below the plot
    ax.set_axisbelow(True)

    # Add figure to dictionary
    dic['precision_vs_th'] = fig
    plt.close(fig)

    # ==== Recall as a function of tau for all provided R ====
    # Create figure
    fig, ax = plt.subplots()
    plt.ioff()
    ax.set_ylabel('Recall')
    ax.set_xlabel(r'$\tau$')
    ax.grid(True)
    plt.title(title)

    list_of_recalls = []

    for r, c in zip(radii, colors):
        # Find closest R
        r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

        # Use only a particular r
        selection = (df.r.values == r_selected) & (df.th.values >= 0)
        if selection.any():
            recall = df.recall.values[selection]
            list_of_recalls.append(recall)
            taus = df.th.values[selection]

            # Plot precision vs tau for this r
            ax.scatter(taus, recall, c=c, s=2, label=f'$r={r}$')

        # Otsu threshold (tau = -1)
        selection = (df.r.values == r_selected) & (df.th.values == -1)
        if selection.any():
            recall = df.recall.values[selection]
            ax.axhline(y=recall,
                       linestyle='-',
                       c=c, label=f'$r={r}$, Otsu')

        # BMM threshold (tau = -2)
        selection = (df.r.values == r_selected) & (df.th.values == -2)
        if selection.any():
            recall = df.recall.values[selection]
            ax.axhline(y=recall,
                       linestyle='--',
                       c=c, label=f'$r={r}$, BMM')


    if len(list_of_recalls) > 0:
        ax.scatter(taus, np.average(np.stack(list_of_recalls), axis=0),
                   c='k', marker='x', s=7, label='avg along $r$')

    # Invert legend order
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]

    # Put legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.03))

    # Hide grid lines below the plot
    ax.set_axisbelow(True)

    # Add figure to dictionary
    dic['recall_vs_tau'] = fig
    plt.close(fig)


    # ==== F-score as a function of tau for all provided R ====
    # Create figure
    fig, ax = plt.subplots()
    plt.ioff()
    ax.set_ylabel('F-score')
    ax.set_xlabel(r'$\tau$')
    ax.grid(True)
    plt.title(title)

    list_of_fscores = []

    for r, c in zip(radii, colors):
        # Find closest R
        r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

        # Use only a particular r
        selection = (df.r.values == r_selected) & (df.th.values >= 0)
        if selection.any():
            fscore = df.fscore.values[selection]
            list_of_fscores.append(fscore)
            taus = df.th.values[selection]

            # Plot precision vs tau for this r
            ax.scatter(taus, fscore, c=c, s=2, label=f'$r={r}$')

        # Otsu threshold (tau = -1)
        selection = (df.r.values == r_selected) & (df.th.values == -1)
        if selection.any():
            fscore = df.fscore.values[selection]
            ax.axhline(y=fscore,
                       linestyle='-',
                       c=c, label=f'$r={r}$, Otsu')

        # BMM threshold (tau = -2)
        selection = (df.r.values == r_selected) & (df.th.values == -2)
        if selection.any():
            fscore = df.fscore.values[selection]
            ax.axhline(y=fscore,
                       linestyle='--',
                       c=c, label=f'$r={r}$, BMM')

    if len(list_of_fscores) > 0:
        ax.scatter(taus, np.average(np.stack(list_of_fscores), axis=0),
                   c='k', marker='x', s=7, label='avg along r')

    # Invert legend order
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]

    # Put legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.03))

    # Hide grid lines below the plot
    ax.set_axisbelow(True)

    # Add figure to dictionary
    dic['fscore_vs_tau'] = fig
    plt.close(fig)

    return dic


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
