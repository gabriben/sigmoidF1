import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class sigmoidF1(nn.Module):

    def __init__(self, S = -1, E = 0):
        super(sigmoidF1, self).__init__()
        self.S = S
        self.E = E

    @torch.cuda.amp.autocast()
    def forward(self, y_hat, y):
        
        y_hat = torch.sigmoid(y_hat)

        b = torch.tensor(self.S)
        c = torch.tensor(self.E)

        sig = 1 / (1 + torch.exp(b * (y_hat + c)))

        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)

        sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - sigmoid_f1
        macroCost = torch.mean(cost)

        return macroCost

class macroSoftF1(nn.Module):

    def __init__(self):
        super(macroSoftF1, self).__init__()

    @torch.cuda.amp.autocast()
    def forward(self, y_hat, y):
        
        y_hat = torch.sigmoid(y_hat)

        tp = torch.sum(y_hat * y, dim=0)
        fp = torch.sum(y_hat * (1 - y), dim=0)
        fn = torch.sum((1 - y_hat) * y, dim=0)

        macroSoft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - macroSoft_f1
        macroCost = torch.mean(cost)

        return macroCost

# https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/loss_functions/losses.py
class CrossEntropyLS(nn.Module):
    def __init__(self, eps: float = 0.2):
        super(CrossEntropyLS, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    @torch.cuda.amp.autocast()
    def forward(self, inputs, target):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)
        cross_entropy_loss_tot = -targets_classes.mul(log_preds)
        cross_entropy_loss = cross_entropy_loss_tot.sum(dim=-1).mean()
        return cross_entropy_loss



# translation from https://github.com/tensorflow/addons/blob/v0.14.0/tensorflow_addons/losses/focal_loss.py#L26-L81
class focalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(focalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_hat, y):
        if self.gamma and self.gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")        

        ceLoss = torch.nn.BCEWithLogitsLoss()
        ce = ceLoss(y, y_hat)

        y_hat = torch.sigmoid(y_hat)

        p_t = (y * y_hat) + ((1 - y) * (1 - y_hat))

        alpha_factor = y * self.alpha + (1 - y) * (1 - self.alpha)
        modulating_factor = torch.pow((1.0 - p_t), self.gamma)

        focal_loss = torch.sum(alpha_factor * modulating_factor * ce)

        return focal_loss

    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    @torch.cuda.amp.autocast()
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    @torch.cuda.amp.autocast()
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    @torch.cuda.amp.autocast()
    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


## from https://github.com/blessu/BalancedLossNLP/blob/main/Reuters/util_loss.py

import numpy as np
import pickle

class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha'] # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        with open("/dbfs/datasets/PASCAL-VOC/VOCasCOCO/class_freq.pkl", 'rb') as f:
            class_freq = pickle.load(f)
        with open("/dbfs/datasets/PASCAL-VOC/VOCasCOCO/train_num.pkl", 'rb') as f:
            train_num = pickle.load(f)
            
        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label==1, self.alpha, 1-self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)             ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None): 
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight
    

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


######## GFM ########

def GFM_loss(y_true, y_pred, n_labels):
    """Custom loss function for the joint estimation of the required parameters for GFM.
    The combined loss is the row-wise sum of categorical losses over the rows of the matrix P
    Where each row corresponds to one label.
    """
    loss = K.constant(0, tf.float32)
    for i in range(n_labels):
        loss += K.categorical_crossentropy(target=y_true[:, i, :],
                                           output=y_pred[:, i, :], from_logits=True)
    return loss

# GFM, not a loss but some postprocessing on the preds
# https://github.com/sdcubber/f-measure/blob/master/src/classifiers/gfm.py

"""
Implementation of the General F-Maximization algorithm
[1] Waegeman, Willem, et al. "On the bayes-optimality of F-measure maximizers." The Journal of Machine Learning Research 15.1 (2014): 3333-3388.
[2] Dembczynski, Krzysztof, et al. "Optimizing the F-measure in multi-label classification: Plug-in rule approach versus structured loss minimization." International Conference on Machine Learning. 2013.
Author: Stijn Decubber
"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)


def labels_to_matrix_Y(y):
    """Convert binary label matrix to a matrix Y that is suitable to estimate P(y,s):
    Each entry of the matrix Y_ij is equal to I(y_ij == 1)*np.sum(yi)"""
    row_sums = np.sum(y, axis=1)
    Y = np.multiply(y, np.broadcast_to(row_sums.reshape(-1, 1), y.shape)).astype(int)
    return(Y)


def labelmatrix_to_GFM_matrix(labelmatrix, max_s):
    """Convert binary labelmatrix to a list that contains for each instance
    a list of n_labels one-hot-encoded vectors"""
    multiclass_matrix = labels_to_matrix_Y(labelmatrix)
    n_instances, n_labels = multiclass_matrix.shape[0], multiclass_matrix.shape[1]

    outputs_per_label = []
    enc = encoder.fit(np.arange(0, max_s + 1).reshape(-1, 1))
    for i in tqdm(range(n_labels)):
        label_i = enc.transform(multiclass_matrix[:, i].reshape(-1, 1))
        outputs_per_label.append(label_i)

    return [np.array([outputs_per_label[i][j, :] for i in range(n_labels)]) for j in range(n_instances)]


def complete_pred(pred, n_labels):
    """Fill up a vector with zeros so that it has length 17."""
    if pred.shape[1] < n_labels:
        pred = np.concatenate(
            (pred, np.zeros(shape=(pred.shape[0], n_labels - pred.shape[1]))), axis=1)
        return(pred)


def complete_matrix_rows(mat):
    # Add rows of zeros to a matrix such that the result has 17 rows
    return np.vstack((mat, np.zeros(shape=(17 - mat.shape[0], mat.shape[1]))))


def complete_matrix_columns_with_zeros(mat, len=17):
    # Add columns of zeros to a matrix such that it has 17 columns
    return np.hstack((mat, np.zeros(shape=(mat.shape[0], len - mat.shape[1]))))


class GeneralFMaximizer(object):
    """ Implementation of the GFM algorithm
    """

    def __init__(self, beta, n_labels):
        self.beta = beta
        self.n_labels = n_labels

    def __matrix_W_F2(self):
        """construct the W matrix for F_beta measure"""
        W = np.ndarray(shape=(self.n_labels, self.n_labels))
        for i in np.arange(1, self.n_labels + 1):
            for j in np.arange(1, self.n_labels + 1):
                W[i - 1, j - 1] = 1 / (i * (self.beta**2) + j)

        return(W)

    def get_predictions(self, predictions):
        """GFM algorithm. Implementation according to [1], page 3528.
        Inputs
        -------
        n_labels: n_labels
        predictions: list of n_instances nparrays that contain *probabilities* required to make up the matrix P
        W: matrix W
        Returns
        ------
        optimal_predictions: F-optimal predictions
        E_f: the expectation of the F-score given x
        """
        # Parameters
        n_instances = len(predictions)
        n_labels = predictions[0].shape[0]  # Each row corresponds to one label

        # Empty containers
        E_F = []
        optimal_predictions = []

        # Set matrix W
        W = self.__matrix_W_F2()

        for instance in range(n_instances):
            # Construct the matrix P

            P = predictions[instance]
            # Compute matrix delta
            D = np.matmul(P, W)

            E = []
            h = []

            for k in range(n_labels):
                # solve inner optimization
                h_k = np.zeros(n_labels)
                # Set h_i=1 to k labels with highest delta_ik
                h_k[np.argsort(D[:, k])[::-1][:k + 1]] = 1
                h.append(h_k)

                # store a value of ...
                E.append(np.dot(h_k, D[:, k]))

            # solve outer maximization problem
            h_F = h[np.argmax(E)]
            E_f = E[np.argmax(E)]

            # Return optimal predictor hF, E[F(Y, hF)]
            optimal_predictions.append(h_F)
            E_F.append(E_f)

        return(np.array(optimal_predictions), E_F)
