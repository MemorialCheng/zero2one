import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


features = pd.read_csv('temps.csv')

# 看看数据长什么样子
features.head()
