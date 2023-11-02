

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False 
import seaborn as sns
import prophet
from prophet.plot import add_changepoints_to_plot
import warnings
warnings.filterwarnings('ignore')

Df = pd.read_excel('/kaggle/input/coal-consume/data.xlsx',parse_dates=["指标名称"])
Df.columns = ['date','amt']
Df.shape

Df.head()

