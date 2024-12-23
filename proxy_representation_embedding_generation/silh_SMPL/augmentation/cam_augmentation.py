import torch


def augment_cam_t(mean_cam_t, xy_std=0.05, delta_z_range=[-5, 5], augment=True):
    """
    Augments camera translation parameters.
    :param mean_cam_t: mean camera translation parameters
    :param xy_std: standard deviation for x and y translation noise
    :param delta_z_range: range for uniform sampling of z translation noise [low, high]
    :param augment: boolean flag to control whether translation augmentation is performed
    :return: augmented camera translation parameters
    """
    batch_size = mean_cam_t.shape[0]
    device = mean_cam_t.device
    new_cam_t = mean_cam_t.clone()
    
    if augment:
        # Add noise to x,y translation
        delta_tx_ty = torch.randn(batch_size, 2, device=device) * xy_std
        new_cam_t[:, :2] = mean_cam_t[:, :2] + delta_tx_ty

        # Add noise to z translation
        l, h = delta_z_range
        delta_tz = (h - l) * torch.rand(batch_size, device=device) + l
        new_cam_t[:, 2] = mean_cam_t[:, 2] + delta_tz
    
    return new_cam_t