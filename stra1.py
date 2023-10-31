from stupids.StupidHead import *
import pandas as pd
from xlwings import view

def doubleMa(*args, **kwargs):
    TradedHQDf = args[0]
    fast_ma = talib.SMA(TradedHQDf.close, timeperiod=kwargs['fast'])
    fast_ma0 = fast_ma[-1]
    fast_ma1 = fast_ma[-2]

    slow_ma = talib.SMA(TradedHQDf.close, timeperiod=kwargs['slow'])
    slow_ma0 = slow_ma[-1]
    slow_ma1 = slow_ma[-2]

    cross_over = fast_ma0 > slow_ma0 and fast_ma1 < slow_ma1
    cross_below = fast_ma0 < slow_ma0 and fast_ma1 > slow_ma1
    if cross_over:
        setpos(1, *args)
    elif cross_below:
        setpos(-1, *args)


if __name__ == '__main__':
    # 修饰原始数据部分
    HQDf = pd.read_csv('data/T888_1d.csv',
                       index_col='date')  # 0 1 2 index换成date这一列作为index
    
    # HQDf = pd.read_csv('data/T888_15m.csv',
    #                    index_col='datetime')  # 0 1 2 index换成date这一列作为index

    HQDf.index = pd.to_datetime(HQDf.index)  # 转成2020-01-02 补0的格式

    ################# part 1
    # # 根据传入的策略进行分析
    ctaParas = {'fast': 5, 'slow': 10}
    ResultTSDf, StatDf = CTA(HQDf, 30, doubleMa, **ctaParas)
    print("-----result1-----")
    print(StatDf)
    # view(ResultTSDf)
    # 绘图部分
    plotResult(ResultTSDf)
    ResultTSDf.to_csv('data/new1.csv')

    # ################# part 2
    # # sapce 是参数空间，定义贝叶斯搜索的空间
    # # func 技术指标名称
    # # fast slow 为技术指标的参数范围
    # space = {
    #         "HQDf": HQDf,
    #         "loadBars": 40,
    #         "func": doubleMa,
    #         "fast": hp.quniform("fast", 3, 30, 1),
    #         "slow": hp.quniform("slow", 5, 40, 1),
    #     }

    # # 调用贝叶斯搜索，第一个参数为参数空间，第二个为优化目标（求解优化目标极值）
    # trials, best = hypeFun(space, 'sharpe_ratio')

    # BestResultTSDf, BestStatDf = CTA(HQDf, 30, doubleMa, **best)
    # print("-----result-----")
    # print(BestStatDf)
    # view(BestResultTSDf)

    # # 绘图部分
    # plotResult(BestResultTSDf)
    # BestResultTSDf.to_csv('data/new2.csv')
