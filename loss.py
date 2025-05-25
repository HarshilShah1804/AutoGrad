def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).sum() * (1.0 / diff.data.size)