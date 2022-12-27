import os
from datetime import date
import pandas as pd
import itertools
import empyrical as empy
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, accuracy_score, confusion_matrix, 
    mean_squared_error, recall_score, f1_score
)

import warnings
warnings.filterwarnings("ignore")


def read_kpi2ret(result_excel_name, start=None, end=None):
    hist = pd.read_excel(result_excel_name, index_col=0, sheet_name='History')
    hist.index = pd.to_datetime(hist.index)
    balance = hist['Balance']
    balance_ = pd.concat([pd.Series(1e9),balance])
    ret = balance_.pct_change().ffill().dropna()
    ret.index = pd.to_datetime(ret.index)
    if start is not None:
        ret = ret[start:]
    if end is not None:
        ret = ret[:end]
    return ret


def cal_metric(y_true, y_pred):

    if isinstance(y_pred, pd.DataFrame):
        prec_ser = pd.Series(index=y_pred.columns, name='precision')
        acccc_ser = pd.Series(index=y_pred.columns, name='accuracy')
        recall_ser = pd.Series(index=y_pred.columns, name='recall')
        recall0_ser = pd.Series(index=y_pred.columns, name='recall0')
        f1_score_ser = pd.Series(index=y_pred.columns, name='f1_score')
        f0_score_ser = pd.Series(index=y_pred.columns, name='f0_score')
        for col in y_pred.columns:
            # print("   *********************************  ", col)
            if isinstance(y_true, pd.DataFrame):
                tmp_y_true = y_true[col]
            else:
                tmp_y_true = y_true.copy()
            prec_ser[col], acccc_ser[col], recall_ser[col], recall0_ser[col], f1_score_ser[col], f0_score_ser[col] = cal_metric(
                tmp_y_true, y_pred[col])
        return prec_ser, acccc_ser, recall_ser, recall0_ser, f1_score_ser, f0_score_ser

    y_true = y_true.reindex(y_pred.index)
    tmp = pd.concat([y_true, y_pred], axis=1).dropna()
    y_true = tmp.iloc[:, 0]
    y_pred = tmp.iloc[:, -1]
    # print("y_pred\n", y_pred)
    # print("y_true\n", y_true)
    # pd.concat([y_proba, y_pred, y_pctch, y_true, ydelta_true], axis=1).to_csv("./tmp/{}.csv".format(y_pctch.name))

    try:
        prec = precision_score(y_true=y_true, y_pred=y_pred)
        acccc = accuracy_score(y_true=y_true, y_pred=y_pred)
    except Exception as e:
        print(e)
        prec = 0
        acccc = 0

    try:
        recall = recall_score(y_true=y_true, y_pred=y_pred)
        recall0 = recall_score(y_true=y_true, y_pred=y_pred, pos_label=0)
    except Exception as e:
        print(e)
        recall = 0
        recall0 = 0

    try:
        f1score = f1_score(y_true=y_true, y_pred=y_pred)
        f0score = f1_score(y_true=y_true, y_pred=y_pred, pos_label=0)
    except Exception as e:
        print(e)
        f1score = 0
        f0score = 0
    # print("mse, mse_proba, msefullscore, msefullscore_ret, prec, acccc, recall, recall0")
    # print(mse, mse_proba, msefullscore, msefullscore_ret, prec, acccc, recall, recall0)
    return prec, acccc, recall, recall0, f1score, f0score


def drawdown_from_price(price):
    if isinstance(price, pd.Series):
        return _drawdown_from_price(price)
    elif isinstance(price, pd.DataFrame):
        return price.apply(_drawdown_from_price)
    else:
        return pd.Series()

def _drawdown_from_price(price):
    shift_max = price.copy()
    _max = price.iloc[0]
    for i, j in price.items():
        #print('i', i)
        #print('j', j)
        _max = max(_max, j)
        shift_max[i] = _max
    return price / shift_max - 1

def return_to_price(ret, ini=100):
    price_0 = ret.dropna().iloc[:1] * 0 + ini
    price_0.index = [0]
    price = (1+ret).cumprod() * ini
    return pd.concat([price_0, price])

def drawdown_from_return(ret, ini=100):
    price = return_to_price(ret, ini)
    return drawdown_from_price(price).iloc[1:]

def avg_drawdown(ret):
    dd = drawdown_from_return(ret)
    return dd.mean()

def empy_metric(ret):
    if isinstance(ret, pd.DataFrame):
        return ret.apply(empy_metric).T
    total_return = lambda x: (1+x).prod()-1
    met_func = [
        total_return, 
        lambda x: empy.annual_return(x), 
        lambda x: empy.sharpe_ratio(x), 
        lambda x: empy.annual_volatility(x), 
        lambda x: empy.max_drawdown(x), 
        avg_drawdown]
    
    met_func_names = ['total_return', 'annual_return', 'sharpe_ratio', 'annual_volatility', 
                      'max_drawdown', 'avg_drawdown',]
    
    se = pd.Series([f(ret) for f in met_func], met_func_names)
    
    se['return/maxdd'] = -se.annual_return/se.max_drawdown
    se['return/avgdd'] = -se.annual_return/se.avg_drawdown
    
    buy01 = ret.apply(lambda x: 0 if x==0 else 1)
    se['buy_ratio'] = buy01.mean()
    se['flip_ratio'] = (buy01-buy01.shift()).abs().mean()

    return se


client_weight_agg = {'sharpe_ratio':5,'annual_return':5, 'max_drawdown':5}
def calculation_score(df, client_weight=client_weight_agg):
    df['Sharpe Score'] = df['sharpe_ratio'].apply(lambda x: x if x>0 else 0) * 100
    df['Return Score'] = df['annual_return'].apply(lambda x: 1+x/2) * 100
    df['Max-DD Score'] = df['max_drawdown'].apply(lambda x: 1+x) * 100

    df['Total Score'] = 0
    for score, weight in client_weight.items():
        df['Total Score'] += weight * df[score]
    return df



# assets = ['DBC', 'EEM', 'IYR', 'TLT', 'VTI', 'SPY', 'QQQ', 'VGK', 'EWJ', 'GLD', 'LQD', 'VYM', 'TIP']
# base_no3vote = './result/MPT_kpi/1.5.1b/'
# mpt_v = '151b'
# start = '2008'
# end = '2022-07'
# client_ret = {}
# for result_excel_name in os.listdir(base_no3vote):
#     if result_excel_name.endswith('.xlsx'):
#         client = result_excel_name.replace('.xlsx', '').split('_')[-1]
#         ret = read_kpi2ret(os.path.join(base_no3vote, result_excel_name), start, end)
#         ret.name = client
#         client_ret[client] = ret


ret_list = [
    './result/ret_corr_threshold_v1-combine2condition(vote_down0).csv',
    './result/ret_corr_threshold_v1.csv'
]

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1")
ret1 = pd.read_csv(ret_list[0], index_col=0)
ret1.index = pd.to_datetime(ret1.index)
ret2 = pd.read_csv(ret_list[1], index_col=0)
ret2.index = pd.to_datetime(ret2.index)
ret2_vote0 = ret2.loc[:, ret2.columns.str.endswith("___vote_down_buy_0")]
ret_vote0 = pd.concat([ret1, ret2_vote0], axis=1)


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2")
for yy in range(2018, 2023):
    print(yy)
    tmp = ret_vote0[str(yy)]
    print(tmp.index[0], tmp.index[-1])
    met = empy_metric(tmp)
    met = calculation_score(met)
    met.to_csv('./result/metrics/{}.csv'.format(yy))


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3")
lookback = [1, 2, 3, 4, 5]
for lb in lookback:
    print("====================== lookback ========================= ", lb)
    range_s = 2008 + lb
    for yy in range(range_s, 2023):
        print("====================== for year ========================= ", yy)
        try:
            start = yy - 5
            end = yy - 1
            print(start, end)
            tmp = ret_vote0[str(start):str(end)]
            print(tmp.index[0], tmp.index[-1])
            met = empy_metric(tmp)
            met = calculation_score(met)
            met.to_csv('./result/metrics/lookback{}_{}-{}.csv'.format(lb, start, end))
        except Exception as e:
            print(e)



