# =============================================================================
# DATA GENERATORS
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

from tensorflow.keras import backend as K


def jaccard_index(y_true, y_pred):
    """Jaccard Index (Interseciton Over Union)
    calculate the Jaccard index between two tensors with the same shape
    Parameters
    ----------
    y_true : ndarray
        ground truth
    y_pred : ndarray
        predictions

    Returns
    -------
    iou : int
        jaccard index
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    dot_product = y_true_flat * y_pred_flat
    intersection = K.sum(dot_product)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection

    iou = intersection / (union + K.epsilon())  # Adding K.epsilon() for numerical stability
    return iou


def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient between two tensors with same shape
    Parameters
    ----------
    y_true : ndarray
        ground truth
    y_pred : ndarray
        predictions

    Returns
    -------
    dice : float
        Dice coefficient
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    dot_product = y_true_flat * y_pred_flat
    intersection = K.sum(dot_product)

    return (2. * intersection) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + K.epsilon())
