import torch
import torch.nn.functional as F

import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def label_to_onehot(target, num_classes=2):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def mixup_criterion(criterion, pred, ys, lam_batch, num_classes):
    k = len(ys)
    if num_classes > 1:
        ys_onehot = [label_to_onehot(y, num_classes=num_classes) for y in ys]
        mixy = vec_mul_ten(lam_batch[:, 0], ys_onehot[0])
        for i in range(1, k):
            mixy += vec_mul_ten(lam_batch[:, i], ys_onehot[i])
        l = cross_entropy_for_onehot(pred, mixy)
    else:
        mixy = vec_mul_ten(lam_batch[:, 0], ys[0])
        for i in range(1, k):
            mixy += vec_mul_ten(lam_batch[:, i], ys[i])
        l = criterion(pred.squeeze(), mixy)
    return l


def vec_mul_ten(vec, tensor):
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res


def mixup(embeds, label, k, embeds_help=None):
    batch_size = embeds.size()[0]
    lams = np.random.normal(0, 1, size=(batch_size, k))
    for i in range(batch_size):
        lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
    lams = torch.from_numpy(lams).float().to(device)
    mixed_x = vec_mul_ten(lams[:, 0], embeds)
    ys = [label]
    # print(lams[:, 0], embeds)
    for i in range(1, k):
        index = torch.randperm(batch_size).to(device)
        if embeds_help is not None and i % 2 == 1:
            mixed_x += vec_mul_ten(lams[:, i], embeds_help[index, :])
        else:
            mixed_x += vec_mul_ten(lams[:, i], embeds[index, :])
        ys.append(label[index])
    return mixed_x, ys, lams
