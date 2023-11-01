# %% 引入包
import pandas as pd
import math

import matplotlib as mpl
import platform
if platform.platform().startswith("Linux"):
    mpl.use("TkAgg")  # Use TKAgg to show figures

import matplotlib.pyplot as plt
import talib  # http://mrjbq7.github.io/ta-lib/doc_index.html
import numpy as np
from sqlalchemy import create_engine
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import importlib
import warnings
import time
from xlwings import view

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

import typing
# %% 自定义函数


def setpos(pos, *args):
    HQDf = args[1]
    idx = args[2]
    HQDf.loc[idx, 'pos'] = pos


def CalculateResult(HQDf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    def get_max_drawdown(array: pd.Series):
        array = pd.Series(array)
        cummax = array.cummax()
        return array / cummax - 1 #当前资金剩余比例除以当前历史cummax 然后 -1 看收益

    HQDf = HQDf.fillna(0) # pos里还有其他的可能为空的填0
    
    HQDf['base_balance'] = HQDf.close / HQDf.close[0]  # 基准净值每天的close除以数据开始的第一天作为盈亏的基准    
    HQDf['chg'] = HQDf.close.pct_change()  # 单日涨跌幅
    # 计算策略净值
    HQDf['strategy_balance'] = 1.0 # 先默认填充1
    for i in range(0, len(HQDf)):
        if i > 0:
            HQDf.loc[HQDf.index[i],'strategy_balance'] = \
                HQDf.iloc[i - 1]['strategy_balance'] * (1. + HQDf.iloc[i]['chg'] * HQDf.iloc[i - 1]['pos'])
            # HQDf.iloc[i]['strategy_balance'] = \
            #     HQDf.iloc[i - 1]['strategy_balance'] * (1. + HQDf.iloc[i]['chg'] * HQDf.iloc[i - 1]['pos'])

    HQDf['drawdown'] = get_max_drawdown(HQDf['strategy_balance'])  # 回撤
    StatDf = {}
    StatDf['MaxDrawDown'] = min(HQDf['drawdown'])  # 最大回撤
    StatDf['return'] = HQDf['strategy_balance'][-1] - 1  # 区间收益
    # 计算年化收益
    years = (HQDf.index[-1] - HQDf.index[0]).days / 365
    if years <= 1:
        StatDf['yearReturn'] = StatDf['return'] / years
    else:
        StatDf['yearReturn'] = (HQDf['strategy_balance'][-1] / 1)**(1 / years) - 1
    StatDf['return/maxdrawdown'] = -1 * StatDf['return'] / StatDf['MaxDrawDown']

    # 计算夏普比
    x = HQDf["strategy_balance"] / HQDf["strategy_balance"].shift(1)
    x[x <= 0] = np.nan
    HQDf["return"] = np.log(x).fillna(0)
    daily_return = HQDf["return"].mean() * 100
    return_std = HQDf["return"].std() * 100
    daily_risk_free = 0.015 / np.sqrt(240)
    StatDf['sharpe_ratio'] = (daily_return - daily_risk_free) / return_std * np.sqrt(240)
    # HQDf = HQDf.dropna()
    return HQDf, StatDf


def plotResult(HQDf: pd.DataFrame) -> typing.NoReturn:
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    HQDf.loc[:, ['base_balance', 'strategy_balance']].plot(ax=axes[0],
                                                           title='净值曲线')
    HQDf.loc[:, ['drawdown']].plot(ax=axes[1], title='回撤', kind='area')
    HQDf.loc[:, ['pos']].plot(ax=axes[2],
                              title='仓位',
                              kind='area',
                              stacked=False)
    HQDf['empty'] = HQDf.close[HQDf.pos == 0]
    HQDf['long'] = HQDf.close[HQDf.pos > 0]
    HQDf['short'] = HQDf.close[HQDf.pos < 0]
    HQDf.loc[:, ['long', 'short', 'empty']].plot(ax=axes[3],
                                                 title='开平仓点位',
                                                 color=["r", "g", "grey"])
    plt.show()


def CTA(HQDf: pd.DataFrame, loadBars: int, strafunc, **kwargs) -> tuple[pd.DataFrame, dict]:
    HQDf['pos'] = np.nan  # 加了一列 pos到最后
    # for idx, hq in tqdm(HQDf.iterrows()):
    for idx, hq in HQDf.iterrows():

        TradedHQDf = HQDf[:idx]  # 每次处理本日前所有数据，递增
        ineridx = TradedHQDf.shape[0]  # 一行内的idx
        if ineridx < loadBars:  # 分析周期，最少多少天
            continue
        strafunc(TradedHQDf, HQDf, idx, ineridx, **kwargs)  # 策略函数处理上面周期的数据
    else:
        HQDf[:idx].pos = HQDf[:idx].pos.fillna(
        method='ffill',inplace=True)  # 本次处理本日前所有的都fillna

    HQDf, StatDf = CalculateResult(HQDf)
    return HQDf, StatDf


def hypeFun(space, target):
    """
    贝叶斯超参数优化
    :param space: 参数空间
    :param target: 优化目标
    :return:
    """

    def hyperparameter_tuning(params):
        HQDf, StatDf = CTA(**params)
        return {"loss": -StatDf[target], "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=hyperparameter_tuning,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    print("hypeFun Best: {}".format(best))
    return trials, best
