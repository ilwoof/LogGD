#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30/07/2022 8:24 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")