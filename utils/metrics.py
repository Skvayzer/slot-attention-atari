import sys

import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F


# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import adjusted_rand_score


class RunningMean:
    def __init__(self):
        self.v = 0.
        self.n = 0

    def update(self, v, n=1):
        self.v += v * n
        self.n += n

    def value(self):
        if self.n:
            return self.v / (self.n)
        else:
            return float('nan')

    def __str__(self):
        return str(self.value())


def adjusted_rand_index(true_mask, pred_mask):
    """Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this function can ignore points in an image
    corresponding to the background (i.e. not to an object).
    Args:
      true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
        The true cluster assignment encoded as one-hot.
      pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
        The predicted cluster assignment encoded as categorical probabilities.
        This function works on the argmax over axis 2.
      name: str. Name of this operation (defaults to "ari_score").
    Returns:
      ARI scores as a Torch `Tensor` of shape [batch_size].
    Raises:
      ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
        We've chosen not to handle the special cases that can occur when you have
        one cluster per datapoint (which would be unusual).
    References:
      Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
        https://link.springer.com/article/10.1007/BF01908075
      Wikipedia
        https://en.wikipedia.org/wiki/Rand_index
      Scikit Learn
        http://scikit-learn.org/stable/modules/generated/\
        sklearn.metrics.adjusted_rand_score.html
    """
    true_mask = true_mask.permute(0, 2, 1)
    pred_mask = pred_mask.permute(0, 2, 1)
    # print("ATTENTION! MASKS (true/pred): ", true_mask.shape, pred_mask.shape, file=sys.stderr, flush=True)
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    # print("ATTA ", n_points, n_true_groups, n_pred_groups, file=sys.stderr, flush=True)

    if (n_points <= n_true_groups and n_points <= n_pred_groups):
        raise ValueError(
        "adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.float()
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).float()

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).float()

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    _all_equal = lambda values: torch.all(torch.eq(values, values[..., :1]), dim=-1)
    both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


# def ari(pred_mask, true_mask, skip_0=False):
#     B = pred_mask.shape[0]
#     pm = pred_mask.argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
#     tm = true_mask.argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
#     aris = []
#     for bi in range(B):
#         t = tm[bi]
#         p = pm[bi]
#         if skip_0:
#             p = p[t > 0]
#             t = t[t > 0]
#         # adjusted_rand_score from sklearn
#         ari_score = adjusted_rand_score(t, p)
#         if ari_score != ari_score:
#             print(f'NaN at bi')
#         aris.append(ari_score)
#     aris = torch.tensor(np.array(aris), device=pred_mask.device)
#     return aris

def mask_iou(
    masks: torch.Tensor,
):
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """
    masks = masks.squeeze()
    B, N, H, W = masks.shape

    masks = masks.view(B, N, H*W)
    iou = 0
    for b in range(B):
        for i in range(N):
            target_mask = masks[b, i]
            print(f"\n\nATTENTION! target mask shape: {target_mask.shape} ", file=sys.stderr, flush=True)

            others_mask = torch.concat((masks[b, :i], masks[b, i+1:]), dim=0).sum(dim=0)
            print(f"\n\nATTENTION! others_mask shape: {others_mask.shape} ", file=sys.stderr, flush=True)

            intersection = others_mask @ target_mask
            print(f"\n\nATTENTION! intersection {intersection} shape: {intersection.shape} ", file=sys.stderr, flush=True)

            area1 = target_mask.sum()
            print(f"\n\nATTENTION! area1: {area1} {area1.shape}", file=sys.stderr, flush=True)

            area2 = others_mask.sum()

            union = (area1.t() + area2) - intersection
            iou += intersection / union

    return iou

def msc(pred_mask, true_mask):
    B = pred_mask.shape[0]
    bpm = pred_mask.argmax(axis=1).squeeze()
    btm = true_mask.argmax(axis=1).squeeze()
    covering = torch.zeros(B, device=pred_mask.device, dtype=torch.float)
    for bi in range(B):
        score = 0.
        norms = 0.
        for ti in range(btm[bi].max()):
            tm = btm[bi] == ti
            if not torch.any(tm): continue
            iou_max = 0.
            for pi in range(bpm[bi].max()):
                pm = bpm[bi] == pi
                if not torch.any(pm): continue
                iou = (tm & pm).to(torch.float).sum() / (tm | pm).to(torch.float).sum()
                if iou > iou_max:
                    iou_max = iou
            r = tm.to(torch.float).sum()
            score += r * iou_max
            norms += r
        covering[bi] = score / norms
    return covering


def reindex(tensor, reindex_tensor, dim=1):
    """
    Reindexes tensor along <dim> using reindex_tensor.
    Effectivelly permutes <dim> for each dimensions <dim based on values in reindex_tensor
    """
    # add dims at the end to match tensor dims.
    alignment_index = reindex_tensor.view(*reindex_tensor.shape,
                                          *([1] * (tensor.dim() - reindex_tensor.dim())))
    return torch.gather(tensor, dim, alignment_index.expand_as(tensor))


def ious_alignment(pred_masks, true_masks):
    tspec = dict(device=pred_masks.device)
    iou_matrix = torch.zeros(pred_masks.shape[0], pred_masks.shape[1], true_masks.shape[1], **tspec)

    true_masks_sums = true_masks.sum((-1, -2, -3))
    pred_masks_sums = pred_masks.sum((-1, -2, -3))

    pred_masks = pred_masks.to(torch.bool)
    true_masks = true_masks.to(torch.bool)

    # Fill IoU row-wise
    for pi in range(pred_masks.shape[1]):
        # Intersection against all cols
        # pandt = (pred_masks[:, pi:pi + 1] * true_masks).sum((-1, -2, -3))
        pandt = (pred_masks[:, pi:pi + 1] & true_masks).to(torch.float).sum((-1, -2, -3))
        # Union against all colls
        # port = pred_masks_sums[:, pi:pi + 1] + true_masks_sums
        port = (pred_masks[:, pi:pi + 1] | true_masks).to(torch.float).sum((-1, -2, -3))
        iou_matrix[:, pi] = pandt / port
        iou_matrix[pred_masks_sums[:, pi] == 0., pi] = 0.

    for ti in range(true_masks.shape[1]):
        iou_matrix[true_masks_sums[:, ti] == 0., :, ti] = 0.

    # NaNs, Inf might come from empty masks (sums are 0, such as on empty masks)
    # Set them to 0. as there are no intersections here and we should not reindex
    iou_matrix = torch.nan_to_num(iou_matrix, nan=0., posinf=0., neginf=0.)

    cost_matrix = iou_matrix.cpu().detach().numpy()
    ious = np.zeros(pred_masks.shape[:2])
    pred_inds = np.zeros(pred_masks.shape[:2], dtype=int)
    for bi in range(cost_matrix.shape[0]):
        true_ind, pred_ind = linear_sum_assignment(cost_matrix[bi].T, maximize=True)
        cost_matrix[bi].T[:, pred_ind].argmax(1)  # Gives which true mask is best for EACH predicted
        ious[bi] = cost_matrix[bi].T[true_ind, pred_ind]
        pred_inds[bi] = pred_ind

    ious = torch.from_numpy(ious).to(pred_masks.device)
    pred_inds = torch.from_numpy(pred_inds).to(pred_masks.device)
    return pred_inds, ious, iou_matrix


def linear_sum_assignment(cost_matrix):
    """Solve the linear sum assignment problem.

    The linear sum assignment problem is also known as minimum weight matching
    in bipartite graphs. A problem instance is described by a matrix C, where
    each C[i,j] is the cost of matching vertex i of the first partite set
    (a "worker") and vertex j of the second set (a "job"). The goal is to find
    a complete assignment of workers to jobs of minimal cost.

    Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is
    assigned to column j. Then the optimal assignment has cost

    .. math::
        \min \sum_i \sum_j C_{i,j} X_{i,j}

    s.t. each row is assignment to at most one column, and each column to at
    most one row.

    This function can also solve a generalization of the classic assignment
    problem where the cost matrix is rectangular. If it has more rows than
    columns, then not every row needs to be assigned to a column, and vice
    versa.

    The method used is the Hungarian algorithm, also known as the Munkres or
    Kuhn-Munkres algorithm.

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.

    Notes
    -----
    .. versionadded:: 0.17.0

    Examples
    --------
    # >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    # >>> from scipy.optimize import linear_sum_assignment
    # >>> row_ind, col_ind = linear_sum_assignment(cost)
    # >>> col_ind
    # array([1, 0, 2])
    # >>> cost[row_ind, col_ind].sum()
    # 5

    References
    ----------
    1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       *Naval Research Logistics Quarterly*, 2:83-97, 1955.

    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       *J. SIAM*, 5(1):32-38, March, 1957.

    5. https://en.wikipedia.org/wiki/Hungarian_algorithm
    """
    cost_matrix = np.asarray(cost_matrix)
    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _Hungary(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    if transposed:
        marked = state.marked.T
    else:
        marked = state.marked
    return np.where(marked == 1)


def average_precision_clevr(pred, attributes, distance_threshold):
  """Computes the average precision for CLEVR.
  This function computes the average precision of the predictions specifically
  for the CLEVR dataset. First, we sort the predictions of the model by
  confidence (highest confidence first). Then, for each prediction we check
  whether there was a corresponding object in the input image. A prediction is
  considered a true positive if the discrete features are predicted correctly
  and the predicted position is within a certain distance from the ground truth
  object.
  Args:
    pred: Tensor of shape [batch_size, num_elements, dimension] containing
      predictions. The last dimension is expected to be the confidence of the
      prediction.
    attributes: Tensor of shape [batch_size, num_elements, dimension] containing
      ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
  Returns:
    Average precision of the predictions.
  """

  [batch_size, _, element_size] = attributes.shape
  [_, predicted_elements, _] = pred.shape

  def unsorted_id_to_image(detection_id, predicted_elements):
    """Find the index of the image from the unsorted detection index."""
    return int(detection_id // predicted_elements)

  flat_size = batch_size * predicted_elements
  flat_pred = np.reshape(pred, [flat_size, element_size])
  sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

  sorted_predictions = np.take_along_axis(
      flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
  idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)

  def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    object_size = np.argmax(target[3:5])
    material = np.argmax(target[5:7])
    shape = np.argmax(target[7:10])
    color = np.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):
    # Extract the current prediction.
    current_pred = sorted_predictions[detection_id, :]
    # Find which image the prediction belongs to. Get the unsorted index from
    # the sorted one and then apply to unsorted_id_to_image function that undoes
    # the reshape.
    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    # Get the ground truth image.
    gt_image = attributes[original_image_idx, :, :]

    # Initialize the maximum distance and the id of the groud-truth object that
    # was found.
    best_distance = 10000
    best_id = None

    # Unpack the prediction by taking the argmax on the discrete attributes.
    (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
     _) = process_targets(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_object_size, target_material, target_shape,
       target_color, target_real_obj) = process_targets(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
        target_attr = [
            target_object_size, target_material, target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          # If a match was found, we check if the distance is below the
          # specified threshold. Recall that we have rescaled the coordinates
          # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
          # `pred_coords`. To compare in the original scale, we thus need to
          # multiply the distance values by 6 before applying the norm.
          distance = np.linalg.norm((target_coords - pred_coords) * 6.)

          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    if best_distance < distance_threshold or distance_threshold == -1:
      # We have detected an object correctly within the distance confidence.
      # If this object was not detected before it's a true positive.
      if best_id is not None:
        if (original_image_idx, best_id) not in detection_set:
          true_positives[detection_id] = 1
          detection_set.add((original_image_idx, best_id))
        else:
          false_positives[detection_id] = 1
      else:
        false_positives[detection_id] = 1
    else:
      false_positives[detection_id] = 1
  accumulated_fp = np.cumsum(false_positives)
  accumulated_tp = np.cumsum(true_positives)
  recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
  precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

  return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
  """Computation of the average precision from precision and recall arrays."""
  recall = recall.tolist()
  precision = precision.tolist()
  recall = [0] + recall + [1]
  precision = [0] + precision + [0]

  for i in range(len(precision) - 1, -0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])

  indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
  ]

  average_precision = 0.
  for i in indices_recall:
    average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
  return average_precision

class Evaluator:
    def __init__(self, masks_have_background=True):
        self.masks_have_background = masks_have_background
        self.stats = defaultdict(RunningMean)
        self.tags = defaultdict(lambda: defaultdict(lambda: defaultdict(RunningMean)))


    def add_statistic(self, name, value, **tags):
        n = 1
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach()
            if len(value.shape):
                n = value.shape[0]
                value = torch.mean(value)
            value = value.item()
        self.stats[name].update(value, n)
        for k, v in tags.items():
            self.tags[name][k][v].update(value, n)

    def statistic(self, name, tag=None):
        if tag is None:
            return self.stats[name].value()
        r = [(k, rm.value()) for k, rm in self.tags[name][tag].items()]
        r = sorted(r, key=lambda x: x[1])
        return r

    @torch.no_grad()
    def update(self,
               pred_image,
               pred_masks,
               true_image,
               true_masks,
               true_metadata=None):
        assert len(pred_image.shape) == 4, "Images should be in (B, C, H, W) shape"

        # TODO: types
        # Coerce pred_masks into known form
        assert 4 <= len(pred_masks.shape) <= 5, "Masks shoudl be in (B, K, 1, H, W) shape"
        pred_masks = pred_masks.view(pred_image.shape[0], -1, 1, *pred_image.shape[-2:])
        total_pred_masks = pred_masks.sum(1, keepdims=True)
        #         assert torch.any(total_pred_masks > 1), "Predicted masks sum out to more than 1."
        if not self.masks_have_background:
            # Some models predict only foreground masks.
            # For convenienve we calculate background masks.
            pred_masks = torch.cat([1. - total_pred_masks, pred_masks], dim=1)

        # Decide the masks Should we effectivelly threshold them?
        K = pred_masks.shape[1]
        pred_masks = pred_masks.argmax(dim=1)
        pred_masks = (pred_masks.unsqueeze(1) == torch.arange(K, device=pred_masks.device).view(1, -1, 1, 1, 1)).to(
            torch.float)
        # Coerce true_Masks into known form
        if len(true_masks.shape) == 4:
            if true_masks.shape[1] == 1:
                # Need to expand into masks
                true_masks = (true_masks.unsqueeze(1) == torch.arange(max(true_masks.max() + 1, pred_masks.shape[1]),
                                                                      device=true_masks.device).view(1, -1, 1, 1,
                                                                                                     1)).to(
                    pred_image.dtype)
            else:
                true_masks = true_masks.unsqueeze(2)
        true_masks = true_masks.view(pred_image.shape[0], -1, 1, *pred_image.shape[-2:])

        K = max(true_masks.shape[1], pred_masks.shape[1])
        if true_masks.shape[1] < K:
            true_masks = torch.cat([true_masks, true_masks.new_zeros(true_masks.shape[0], K - true_masks.shape[1], 1,
                                                                     *true_masks.shape[-2:])], dim=1)
        if pred_masks.shape[1] < K:
            pred_masks = torch.cat([pred_masks, pred_masks.new_zeros(pred_masks.shape[0], K - pred_masks.shape[1], 1,
                                                                     *pred_masks.shape[-2:])], dim=1)

        mse = F.mse_loss(pred_image, true_image, reduction='none').sum((1, 2, 3))
        self.add_statistic('MSE', mse)

        # If argmax above, these masks are either 0 or 1
        pred_count = (pred_masks >= 0.5).any(-1).any(-1).any(-1).to(torch.float).sum(-1)  # shape: (B,)
        true_count = (true_masks >= 0.5).any(-1).any(-1).any(-1).to(torch.float).sum(-1)  # shape: (B,)
        accuracy = (true_count == pred_count).to(torch.float)
        self.add_statistic('acc', accuracy)

        pred_reindex, ious, _ = self.ious_alignment(pred_masks, true_masks)
        pred_masks = self.reindex(pred_masks, pred_reindex, dim=1)

        truem = true_masks.any(-1).any(-1).any(-1)
        predm = pred_masks.any(-1).any(-1).any(-1)

        vism = truem | predm
        num_pairs = vism.to(torch.float).sum(-1)

        # mIoU
        mIoU = ious.sum(-1) / num_pairs
        mIoU_fg = ious[:, 1:].sum(-1) / (num_pairs - 1)  # do not consider the background
        mIoU_gt = ious.sum(-1) / truem.to(torch.float).sum(-1)

        self.add_statistic('mIoU', mIoU)
        self.add_statistic('mIoU_fg', mIoU_fg)
        self.add_statistic('mIoU_gt', mIoU_gt)

        msc = self.msc(pred_masks, true_masks)
        self.add_statistic('mSC', msc)

        # DICE
        dices = 2 * (pred_masks * true_masks).sum((-3, -2, -1)) / (
                pred_masks.sum((-3, -2, -1)) + true_masks.sum((-3, -2, -1)))
        dices = torch.nan_to_num(dices, nan=0., posinf=0.)  # if there were any empties, they now have 0. DICE

        dice = dices.sum(-1) / num_pairs
        dice_fg = dices[:, 1:].sum(-1) / (num_pairs - 1)
        self.add_statistic('DICE', dice)
        self.add_statistic('DICE_FG', dice_fg)

        # ARI
        ari = self.ari(pred_masks, true_masks)
        ari_fg = self.ari(pred_masks, true_masks, skip_0=True)
        if torch.any(torch.isnan(ari_fg)):
            print('NaN ari_fg')
        if torch.any(torch.isinf(ari_fg)):
            print('Inf ari_fg')
        self.add_statistic('ARI', ari)
        self.add_statistic('ARI_FG', ari_fg)

        # mAP --?

        if true_metadata is not None:
            smses = F.mse_loss(pred_image[:, None] * true_masks,
                               true_image[:, None] * true_masks, reduction='none').sum((-1, -2, -3))

            for bi, meta in enumerate(true_metadata):
                # ground
                self.add_statistic('ground_mse', smses[bi, 0], ground_material=meta['ground_material'])
                self.add_statistic('ground_iou', ious[bi, 0], ground_material=meta['ground_material'])

                for i, obj in enumerate(meta['objects']):
                    tags = {k: v for k, v in obj.items() if k != 'rotation'}
                    if truem[bi, i + 1]:
                        self.add_statistic('obj_mse', smses[bi, i + 1], **tags)
                        self.add_statistic('obj_iou', ious[bi, i + 1], **tags)
                        # Maybe number of components?
        return pred_masks, true_masks


class _Hungary(object):
    """State of the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Must have shape[1] >= shape[0].
    """

    def __init__(self, cost_matrix):
        self.C = cost_matrix.copy()

        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=bool)
        self.col_uncovered = np.ones(m, dtype=bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= np.asarray(state.col_uncovered, dtype=int)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    np.asarray(state.row_uncovered, dtype=int))
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4
