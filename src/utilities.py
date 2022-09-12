#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 19/05/2022 10:34 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""

import numpy as np
import torch
from src.configuration import device

# calculateMahalanobis function to calculate
# the Mahalanobis distance
def calculateMahalanobis(y=None, data=None, cov=None):
    y_mu = y - torch.mean(data)
    if not cov:
        cov = torch.cov(data.T)
    inv_covmat = torch.linalg.inv(cov)
    left = torch.dot(y_mu, inv_covmat)
    mahal = torch.dot(left, y_mu.T)
    return mahal.diagonal()

def init_center_c(input_dim, train_loader, model, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(input_dim, device=device)

    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(torch.sigmoid(outputs.squeeze()), dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    print(f"Initialize c successfully")

    return c

def get_radius(dist, nu):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

