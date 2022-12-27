import argparse
import os
import itertools

import numpy as np
import pandas as pd
import talib as ta
import parm_list



raw_data_folder = 'data'



def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--asset', help='asset',
                        type=str, default='')
    args = vars(parser.parse_args())
    return args


def get_price(file):
    price = pd.read_csv(file, index_col=0)
    price.index = pd.to_datetime(price.index)
    price = price.sort_index()
    price = price.ffill()
    return price


def gen_ema_feature(price, span):
    # ewma_func = lambda x, y: x.ewm(span=y).mean()
    # fast_indicator = ewma(price.shift(), span)
    ema = price.shift().ewm(span=span).mean().dropna()
    ema.name = '{}_price_EMA{}'.format(setting['asset'], span)
    return ema


def gen_ema_return(price, span):
    # ewma_func = lambda x, y: x.ewm(span=y).mean()
    # fast_indicator = ewma(price.shift(), span)
    ema = price.pct_change().shift().ewm(span=span).mean().dropna()
    ema.name = '{}_return_EMA{}'.format(setting['asset'], span)
    return ema


def gen_cross_feature(iterable, r, feature):
    cross_feature = {} #pd.DataFrame()
    for subset in itertools.combinations(iterable, r):
        #print("subset", subset)
        key_a = "{}_price_EMA{}".format(setting['asset'], subset[0])
        key_b = "{}_price_EMA{}".format(setting['asset'], subset[1])
        groupname = "{}&{}".format(key_a, key_b)

        feature_a = feature[key_a]
        feature_b = feature[key_b]

        new_fea = feature_a / feature_b
        new_fea.name = "{}/{}".format(key_a, key_b)
        new_fea2 = feature_b / feature_a
        new_fea2.name = "{}/{}".format(key_b, key_a)
        cross_feature[groupname] = pd.concat([new_fea, new_fea2], axis=1)
        #print("=============================\n{}:\n{}".format(groupname, cross_feature[groupname]))
    return cross_feature


def gen_extra_feature(raw_df):
    extra_feature = {} #pd.DataFrame()
    
    shifted_df = raw_df.shift().dropna()
    
    volume_name = '{}_volume'.format(setting['asset'])
    price__div__prev_max_name = '{}_price__div__prev_max'.format(setting['asset'])
    dd_duration_name = '{}_dd_duration'.format(setting['asset'])
    
    try:
        extra_feature[volume_name] = shifted_df.volume.rename(volume_name).to_frame()
    except:
        print()
        pass
    price_shift = shifted_df['adj_close']
    
    prev_max = pd.Series()
    dd_duration = pd.Series()
    _max = price_shift.iloc[0]
    _ddd = 0
    for i, j in price_shift.items():
        _max = max(_max, j)
        _ddd = _ddd+1 if _max>j else 0
        prev_max.loc[i] = _max
        dd_duration.loc[i] = _ddd
    
    price__div__prev_max = price_shift / prev_max
    price__div__prev_max = price__div__prev_max.rename(price__div__prev_max_name).to_frame()
    
    #price__div__prev_max_pct_change = price__div__prev_max.pct_change()
    #price__div__prev_max_pct_change.columns = ['price__div__prev_max_pct_change']
    
    extra_feature[price__div__prev_max_name] = price__div__prev_max
    #extra_feature['price__div__prev_max_pct_change'] = price__div__prev_max_pct_change
    extra_feature[dd_duration_name] = dd_duration.rename(dd_duration_name).to_frame()
    
    return extra_feature


def gen_price_position(price, lookback=[5, 10]):
    shift_price = price.shift().dropna()
    price_pos = pd.DataFrame()
    for lb in lookback:
        # print("===========================================: ", type(lb), lb)
        max_price = shift_price.rolling(lb).apply(max).dropna()
        # print("max_price: ", type(max_price), "\n", max_price.head(10))
        min_price = shift_price.rolling(lb).apply(min).dropna()
        # print("min_price: ", type(min_price), "\n", min_price.head(10))
        price_pos_sub = (shift_price - min_price) / (max_price - min_price)
        price_pos_sub.name = 'price_position_lookback{}D'.format(lb)
        price_pos_sub = price_pos_sub.dropna()
        # print("price_pos_sub: ", type(price_pos_sub), "\n", price_pos_sub.head())
        price_pos = pd.concat([price_pos, price_pos_sub], axis=1)
    # print("price_pos\n", price_pos)
    return price_pos


def period_return(df):
    pct_ser = pd.Series()
    for i in range(1, len(df)):
        sub_pct = df.pct_change(i)
        sub_pct = sub_pct.dropna()
        pct_ser = pd.concat([pct_ser, sub_pct[[0]]])
    return pct_ser


def gen_pct_period_start(price, period):
    shift_price = price.shift().dropna()
    datelist = pd.Series(pd.to_datetime(shift_price.index), index=pd.to_datetime(shift_price.index))
    datelist = datelist.sort_index()
    period_index = datelist.index.to_period(period)

    period_list = list(set(period_index))
    period_list.sort()

    pct_period_start = pd.Series()
    for sub_period in period_list:
        sub_df = shift_price[sub_period.start_time.date():sub_period.end_time.date()]
        pct_df = period_return(sub_df)
        pct_period_start = pd.concat([pct_period_start, pct_df])
    return pct_period_start


def period_MA(df):
    # print(df)
    ma_ser = pd.Series()
    for i in range(1, len(df)+1):
        sub_ma = ta.MA(df, timeperiod=i, matype=0)
        sub_ma = sub_ma.dropna()
        # print("----------------------{}----------------------\n{}".format(i, sub_ma))
        ma_ser = pd.concat([ma_ser, sub_ma[[0]]])
    return ma_ser


def gen_ma_period_start(price, period):
    shift_price = price.shift().dropna()
    datelist = pd.Series(pd.to_datetime(shift_price.index), index=pd.to_datetime(shift_price.index))
    datelist = datelist.sort_index()
    period_index = datelist.index.to_period(period)

    period_list = list(set(period_index))
    period_list.sort()

    ma_period_start = pd.Series()
    for sub_period in period_list:
        sub_df = shift_price[sub_period.start_time.date():sub_period.end_time.date()]
        ma_ser = period_MA(sub_df)
        ma_period_start = pd.concat([ma_period_start, ma_ser])
    return ma_period_start


def gen_delta_forward(price, predict_period, boolean=True):
    '''
    df = price.reset_index(drop=False)
    num = len(price)
    date_list = df['Date'][:num - predict_period]
    df_new = pd.DataFrame()
    df_new['Date'] = date_list
    delta_price = []
    for i in range(num - predict_period):
        predict_price = price[i + predict_period]
        current_price = price[i]
        delta = (predict_price - current_price) / current_price
        delta_price.append(delta)

    feature_name = 'after_%ddays_delta' % predict_period
    
    print(feature_name)
    
    df_new[feature_name] = delta_price

    df_labels = df_new.set_index(['Date'])
    p_delta = df_labels.iloc[:, 0].values
    label = (p_delta > 0) * 1
    df = pd.DataFrame(data=label, index=df_labels.index, columns=['label'])
    '''
    # .pct_change()             comp n     form n-1
    # .pct_change(k)            comp n     form n-k
    # .pct_change(k).shift(1-k) comp n+k-1 form n-1
    se = price.pct_change(predict_period).shift(1-predict_period).dropna()
    if boolean:
        se = (se >= 0)*1
    df = se.to_frame()
    return df


def ret_df_targetname(assets):
    '''
    Load the raw dataset by name of the asset
    return the the dataframe of the asset's quote
    and name for the csv of the TA
    '''
    fname = os.path.join(raw_data_folder, '%s.csv' %assets)
    df = pd.read_csv(fname)
    try:
        df['volume'] = df['volume'] / 1.0
    except:
        pass

    target_name = os.path.join(ta_feature_folder, '%s_delta.csv' % assets)

    return df, target_name





def call_rsi(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    close=df["close"].values
    lennn=len(target['Pct_Change_1'].values)
    sub_feature = pd.DataFrame(index=df['date'])
    for day in day_range:
        real=ta.RSI(close,day)
        col_name="RSI_%d"%day
        len_real=len(real)
        sub_feature[col_name]=real[len_real-lennn:len_real]
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)
    sub_feature = sub_feature.dropna(how='all')
    sub_feature.to_csv(os.path.join(ta_feature_folder, '%s_rsi.csv' % assets))

def call_macd(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    close=df["close"].values
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=parm[i][0], slowperiod=parm[i][1], signalperiod=parm[i][2])
        col_name="MACD_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(macd)
        target[col_name+"_macd"]=macd[len_real-lennn:len_real]
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=parm[i][0], slowperiod=parm[i][1], signalperiod=parm[i][2])
        col_name="MACD_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(macd)
        target[col_name+"_macdsignal"]=macdsignal[len_real-lennn:len_real]
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=parm[i][0], slowperiod=parm[i][1], signalperiod=parm[i][2])
        col_name="MACD_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(macd)
        target[col_name+"_macdhist"]=macdhist[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_obv(assets):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    close=df["close"].values
    vol=df["volume"].values
    real=ta.OBV(close,vol)
    len_real=len(real)
    target["OBV"]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_atr(assets,day_range):
    df,target_name = ret_df_targetname(assets)

    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)

    for i in day_range:
        col_name="ATR_%s"%i
        real = ta.ATR(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)


def call_bband(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)

    for i in range(len(parm)):
        upperband, middleband, lowerband = ta.BBANDS(close, timeperiod=parm[i][0], nbdevup=parm[i][1], nbdevdn=parm[i][2], matype=parm[i][3])
        col_name="BBANDS"
        up_name=col_name+"_upperband"+"_%s_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2],parm[i][3])
        len_real=len(upperband)
        target[up_name]=upperband[len_real-lennn:len_real]
    for i in range(len(parm)):
        upperband, middleband, lowerband = ta.BBANDS(close, timeperiod=parm[i][0], nbdevup=parm[i][1], nbdevdn=parm[i][2], matype=parm[i][3])
        col_name="BBANDS"
        mid_name=col_name+"_middleband"+"_%s_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2],parm[i][3])
        len_real=len(upperband)
        target[mid_name]=middleband[len_real-lennn:len_real]
    for i in range(len(parm)):
        upperband, middleband, lowerband = ta.BBANDS(close, timeperiod=parm[i][0], nbdevup=parm[i][1], nbdevdn=parm[i][2], matype=parm[i][3])
        col_name="BBANDS"
        low_name=col_name+"_lowerband"+"_%s_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2],parm[i][3])
        len_real=len(upperband)
        target[low_name]=lowerband[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)


def call_ad(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    volume=df["volume"].values
    target=pd.read_csv(target_name)
    real = ta.AD(high, low, close, volume)
    lennn=len(target['Pct_Change_1'].values)
    len_real=len(real)
    target["AD"]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_mfi(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    volume=df["volume"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real = ta.MFI(high, low, close, volume,parm[i])
        col_name="MFI_%s" %parm[i]
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_kama(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.KAMA(close,i)
        col_name="KAMA_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_sma_vol(assets,parm):
    df,target_name = ret_df_targetname(assets)
    volume=df["volume"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.SMA(volume,i)
        col_name="SMA_VOL_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_stochf(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        fastk, fastd = ta.STOCHF(high, low, close, fastk_period=parm[i][0], fastd_period=parm[i][1])
        col_name="stochf"
        k_name=col_name+"_fastk"+"_%s_%s"%(parm[i][0],parm[i][1])
        len_real=len(fastd)
        target[k_name]=fastk[len_real-lennn:len_real]
    for i in range(len(parm)):
        fastk, fastd = ta.STOCHF(high, low, close, fastk_period=parm[i][0], fastd_period=parm[i][1])
        col_name="stochf"
        d_name=col_name+"_fastd"+"_%s_%s"%(parm[i][0],parm[i][1])
        len_real=len(fastd)
        target[d_name]=fastd[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_ad_oscil(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values
    close=df["close"].values
    vol=df["volume"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real  = ta.ADOSC(high, low, close,vol, fastperiod=parm[i][0], slowperiod=parm[i][1])
        col_name="ad_oscil"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_mom_slope(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        tmp=[]
        for j in range(len(df["close"])-(parm[i])):
            h=df["close"]
            mom=h.iloc[-(parm[i]+j):-(1+j)].mean() / h.iloc[-((parm[i]+j)+1):-(2+j)].mean() -1
            mom_n = mom / np.sum(abs(mom))
            tmp.append(mom_n)
        for j in range(parm[i]):
            tmp.append(np.nan)
        real=tmp[::-1]
        len_real=len(real)
        col_name="mom_slope_"+str(parm[i])
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_mom_z(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        tmp=[]
        for j in range(len(df["close"])-(parm[i])):
            h=df["close"]
            mom = (h.iloc[-1] - h.iloc[-parm[i]:-1].mean())/h.iloc[-parm[i]:-1].std()
            mom_n = mom / np.sum(abs(mom))
            tmp.append(mom_n)
        for j in range(parm[i]):
            tmp.append(np.nan)
        real=tmp[::-1]
        len_real=len(real)
        col_name="mom_z_"+str(parm[i])
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_mom_p_now(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        tmp=[]
        for j in range(len(df["close"])-(parm[i])):
            h=df["close"]
            mom =h.iloc[-(1+j)]/h.iloc[-(parm[i]+j)] - 1
            mom_n = mom / np.sum(abs(mom))
            tmp.append(mom_n)
        for j in range(parm[i]):
            tmp.append(np.nan)
        real=tmp[::-1]
        len_real=len(real)
        col_name="mom_p_now_"+str(parm[i])
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_mom_SMA_SLCross(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        n1=parm[i][0]
        n2=parm[i][1]
        h=df["volume"]
        if n1 ==1:
            sma_short=h
        else:
            sma_short=ta.SMA(pd.np.array(h), parm[i][0])
        sma_long = ta.SMA(pd.np.array(h), parm[i][1])

        mom = sma_short / sma_long -1
        abs_mom=abs(mom)
        abs_mom=np.nan_to_num(abs_mom)
        sum_abs_mom=np.sum(abs_mom)
        mom_n = mom / sum_abs_mom
        real=mom_n
        len_real=len(real)
        col_name="mom_SMA_SLCross_%s_%s"%(n1,n2)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)


def call_mom_EMA_SLCross(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        h=df["volume"]
        n1=parm[i][0]
        n2=parm[i][1]
        if n1 ==1:
            sma_short=h
        else:
            sma_short=ta.EMA(pd.np.array(h), parm[i][0])
        sma_long = ta.EMA(pd.np.array(h), parm[i][1])
        mom = sma_short / sma_long -1
        abs_mom=abs(mom)
        abs_mom=np.nan_to_num(abs_mom)
        sum_abs_mom=np.sum(abs_mom)
        mom_n = mom / sum_abs_mom
        real=mom_n
        len_real=len(real)
        col_name="mom_EMA_SLCross_%s_%s"%(n1,n2)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_mom_WMA_SLCross(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        h=df["volume"]
        n1=parm[i][0]
        n2=parm[i][1]
        if n1 ==1:
            sma_short=h
        else:
            sma_short=ta.WMA(pd.np.array(h), parm[i][0])
        sma_long = ta.WMA(pd.np.array(h), parm[i][1])

        mom = sma_short / sma_long -1
        abs_mom=abs(mom)
        abs_mom=np.nan_to_num(abs_mom)
        sum_abs_mom=np.sum(abs_mom)
        mom_n = mom / sum_abs_mom
        real=mom_n
        len_real=len(real)
        col_name="mom_WMA_SLCross_%s_%s"%(n1,n2)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_mom_Pct_chg_SLCross(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        h=df["volume"]
        n1=parm[i][0]
        n2=parm[i][1]
        sma_short=ta.ROC(pd.np.array(h), parm[i][0])
        sma_long = ta.ROC(pd.np.array(h), parm[i][1])

        mom = sma_short / sma_long -1
        abs_mom=abs(mom)
        abs_mom=np.nan_to_num(abs_mom)
        sum_abs_mom=np.sum(abs_mom)
        mom_n = mom / sum_abs_mom
        real=mom_n
        len_real=len(real)
        col_name="mom_ROC_SLCross_%s_%s"%(n1,n2)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)



def call_sma_now_cross(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        tmp=[]
        for j in range(len(df["close"])-(parm[i])):
            h=df["close"]
            price_now = h.iloc[-(1+j)]
            sma_long = h.iloc[-(parm[i]+j):-(1+j)].mean()
            mom = price_now / sma_long -1
            mom_n = mom / np.sum(abs(mom))
            tmp.append(mom_n)
        for j in range(parm[i]):
            tmp.append(np.nan)
        real=tmp[::-1]
        len_real=len(real)
        col_name="sma_now_cross_"+str(parm[i])
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_willr(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real = ta.WILLR(high, low, close,parm[i])
        col_name="WILLR_%s" %parm[i]
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_rocr(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.ROCR(close,i)
        col_name="ROCR_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_mom(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.MOM(close,i)
        col_name="MOM_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cci(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="CCI_%s"%i
        real = ta.CCI(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_adx(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="ADX_%s"%i
        real = ta.ADX(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_trix(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="TRIX_%s"%i
        real = ta.TRIX(close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_tsf(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="TSF_%s"%i
        real = ta.TSF(close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_sma(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        #print i
        real=ta.SMA(close,i)
        col_name="SMA_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_ema(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        #print i
        real=ta.EMA(close,i)
        col_name="EMA_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_dema(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        #print i
        real=ta.DEMA(close,i)
        col_name="DEMA_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_ht_trendline(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    #print i
    real=ta.HT_TRENDLINE(close)
    col_name="HT_TRENDLINE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_MA(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real = ta.MA(close, timeperiod=parm[i][0], matype=parm[i][1])
        col_name="MA"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        #print k_name
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_MAVP(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real = ta.MAVP(close, periods=df.date, matype=parm[i][1])
        col_name="MAVP"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        #print k_name
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_mama(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)

    mama, fama= ta.MAMA(close)
    col_name="MAMA"
    k_name=col_name+"_mama"
    len_real=len(mama)
    target[k_name]=mama[len_real-lennn:len_real]

    mama, fama = ta.MAMA(close)
    col_name="MAMA"
    d_name=col_name+"_fama"
    len_real=len(fama)
    target[d_name]=fama[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_midpoint(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        #print i
        real=ta.MIDPOINT(close,i)
        col_name="MIDPOINT_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_midprice(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        #print i
        real=ta.MIDPRICE(high,low,i)
        col_name="MIDPRICE_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_sar(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real  = ta.SAR(high, low, acceleration=parm[i][0], maximum=parm[i][1])
        col_name="SAR"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        k_name=k_name.replace(".","")
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_sarext(assets):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)

    val= ta.SAREXT(high,low)
    col_name="SAREXT"
    len_real=len(val)
    target[col_name]=val[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_t3(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real =ta.T3( close, timeperiod=parm[i][0], vfactor=parm[i][1])
        col_name="T3"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_tema(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="TEMA_%s"%i
        real = ta.TEMA(close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_trima(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="TRIMA_%s"%i
        real = ta.TRIMA(close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_wma(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="WMA_%s"%i
        real = ta.WMA(close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_adxr(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="ADXR_%s"%i
        real = ta.ADXR(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_apo(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    close=df["close"].values
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real = ta.APO(close, fastperiod=parm[i][0], slowperiod=parm[i][1], matype=parm[i][2])
        col_name="APO_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_arron(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        aroondown,aroonup=ta.AROON(high, low, timeperiod=parm[i])
        col_name="AROON"
        k_name=col_name+"_aroondown"+"_%s"%(parm[i])
        len_real=len(aroondown)
        target[k_name]=aroondown[len_real-lennn:len_real]

    for i in range(len(parm)):
        aroondown,aroonup=ta.AROON(high, low, timeperiod=parm[i])
        col_name="AROON"
        d_name=col_name+"_aroonup"+"_%s"%(parm[i])
        len_real=len(aroonup)
        target[d_name]=aroonup[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)


def call_aroonosc(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.AROONOSC(high,low,timeperiod=i)
        col_name="AROONOSC_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_bop(assets):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    close=df["close"].values
    open=df["open"].values
    high=df["high"].values
    low=df["low"].values
    real=ta.BOP(open, high, low, close)
    len_real=len(real)
    target["BOP"]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cmo(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="CMO_%s"%i
        real = ta.CMO(close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_dx(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="DX_%s"%i
        real = ta.DX(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_macdext(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    close=df["close"].values
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACDEXT(close, fastperiod=parm[i][0], slowperiod=parm[i][1], signalperiod=parm[i][2])
        col_name="MACDEXT_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(macd)
        target[col_name+"_macd"]=macd[len_real-lennn:len_real]
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACDEXT(close, fastperiod=parm[i][0], slowperiod=parm[i][1], signalperiod=parm[i][2])
        col_name="MACDEXT_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(macd)
        target[col_name+"_macdsignal"]=macdsignal[len_real-lennn:len_real]
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACDEXT(close, fastperiod=parm[i][0], slowperiod=parm[i][1], signalperiod=parm[i][2])
        col_name="MACDEXT_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(macd)
        target[col_name+"_macdhist"]=macdhist[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_macdfix(assets,parm):
    df,target_name = ret_df_targetname(assets)
    target=pd.read_csv(target_name)
    close=df["close"].values
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACDFIX(close ,signalperiod=parm[i])
        col_name="MACDFIX_%s"%(parm[i])
        len_real=len(macd)
        target[col_name+"_macd"]=macd[len_real-lennn:len_real]
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACDFIX(close,signalperiod=parm[i])
        col_name="MACDFIX_%s"%(parm[i])
        len_real=len(macd)
        target[col_name+"_macdsignal"]=macdsignal[len_real-lennn:len_real]
    for i in range(len(parm)):
        macd, macdsignal, macdhist = ta.MACDFIX(close,signalperiod=parm[i])
        col_name="MACDFIX_%s"%(parm[i])
        len_real=len(macd)
        target[col_name+"_macdhist"]=macdhist[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_minus_di(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="MINUS_DI_%s"%i
        real = ta.MINUS_DI(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_minus_dm(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="MINUS_DM_%s"%i
        real = ta.MINUS_DM(high, low, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_plus_di(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="PLUS_DI_%s"%i
        real = ta.PLUS_DI(high, low, close, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_plus_dm(assets,day_range):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in day_range:
        col_name="PLUS_DM_%s"%i
        real = ta.PLUS_DM(high, low, timeperiod=i)
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_ppo(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        val= ta.PPO(close, fastperiod=parm[i][0], slowperiod=parm[i][1], matype=parm[i][2])
        col_name="PPO"
        up_name=col_name+"_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(val)
        target[up_name]=val[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_rocp(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.ROCP(close,i)
        col_name="ROCP_%d"%i
        len_real=len(real)
        target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_stoch_slow(assets,parm):
    df,target_name = ret_df_targetname(assets)
    high=df["high"].values
    low=df["low"].values
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=parm[i][0], slowk_period=parm[i][1], slowd_period=parm[i][2])
        col_name="stoch"
        k_name=col_name+"_slowk"+"_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(slowk)
        target[k_name]=slowk[len_real-lennn:len_real]

    for i in range(len(parm)):
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=parm[i][0], slowk_period=parm[i][1], slowd_period=parm[i][2])
        col_name="stoch"
        d_name=col_name+"_slowd"+"_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(slowd)
        target[d_name]=slowd[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_stocrsi(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        fastk, fastd = ta.STOCHRSI(close, timeperiod=parm[i][0], fastk_period=parm[i][1], fastd_period=parm[i][2])
        col_name="stocrsi"
        k_name=col_name+"_fastk"+"_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(fastk)
        target[k_name]=fastk[len_real-lennn:len_real]

    for i in range(len(parm)):
        fastk, fastd = ta.STOCHRSI(close, timeperiod=parm[i][0], fastk_period=parm[i][1], fastd_period=parm[i][2])
        col_name="stocrsi"
        d_name=col_name+"_fastd"+"_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(fastd)
        target[d_name]=fastd[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_ultosc(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.ULTOSC(high,low,close, timeperiod1=parm[i][0], timeperiod2=parm[i][1], timeperiod3=parm[i][2])
        col_name="ultosc"
        k_name=col_name+"_%s_%s_%s"%(parm[i][0],parm[i][1],parm[i][2])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_natr(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.NATR(high,low,close, timeperiod=parm[i])
        col_name="NATR"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_trange(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.TRANGE(high,low,close)
    col_name="TRANGE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_ht_dcperiod(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.HT_DCPERIOD(close)
    col_name="HT_DCPERIOD"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_ht_dcphase(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.HT_DCPHASE(close)
    col_name="HT_DCPHASE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_ht_phasor(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)

    inphase, quadrature = ta.HT_PHASOR(close)
    col_name="ht_phasor"
    k_name=col_name+"_inphase"
    len_real=len(inphase)
    target[k_name]=inphase[len_real-lennn:len_real]

    inphase, quadrature = ta.HT_PHASOR(close)
    col_name="ht_phasor"
    d_name=col_name+"_quadrature"
    len_real=len(quadrature)
    target[d_name]=quadrature[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_ht_sine(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)

    sine, leadsine = ta.HT_SINE(close)
    col_name="ht_sine"
    k_name=col_name+"_sine"
    len_real=len(sine)
    target[k_name]=sine[len_real-lennn:len_real]

    sine, leadsine = ta.HT_SINE(close)
    col_name="ht_sine"
    d_name=col_name+"_leadsine"
    len_real=len(leadsine)
    target[d_name]=leadsine[len_real-lennn:len_real]

    target.to_csv(target_name, index = None)

def call_ht_trendmode(assets):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.HT_TRENDMODE(close)
    col_name="HT_TRENDMODE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_avgprice(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.AVGPRICE(open,high,low,close)
    col_name="AVGPRICE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_medprice(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.MEDPRICE(high,low)
    col_name="MEDPRICE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_typprice(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.TYPPRICE(high,low,close)
    col_name="TYPPRICE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_wclprice(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.WCLPRICE(high,low,close)
    col_name="WCLPRICE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl2crows(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL2CROWS(open,high,low,close)
    col_name="CDL2CROWS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl3blackcrows(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL3BLACKCROWS(open,high,low,close)
    col_name="CDL3BLACKCROWS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl3inside(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL3INSIDE(open,high,low,close)
    col_name="CDL3INSIDE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl3linestrike(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL3LINESTRIKE(open,high,low,close)
    col_name="CDL3LINESTRIKE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl3outside(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL3OUTSIDE(open,high,low,close)
    col_name="CDL3OUTSIDE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl3starsinsouth(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL3STARSINSOUTH(open,high,low,close)
    col_name="CDL3STARSINSOUTH"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdl3whitesoldiers(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDL3WHITESOLDIERS(open,high,low,close)
    col_name="CDL3WHITESOLDIERS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlabandonedbaby(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLABANDONEDBABY(open,high,low,close)
    col_name="CDLABANDONEDBABY"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdladvanceblock(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLADVANCEBLOCK(open,high,low,close)
    col_name="CDLADVANCEBLOCK"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlbelthold(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLBELTHOLD(open,high,low,close)
    col_name="CDLBELTHOLD"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlbreakaway(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLBREAKAWAY(open,high,low,close)
    col_name="CDLBREAKAWAY"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlclosingmarubozu(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLCLOSINGMARUBOZU(open,high,low,close)
    col_name="CDLCLOSINGMARUBOZU"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlconcealbabyswall(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLCONCEALBABYSWALL(open,high,low,close)
    col_name="CDLCONCEALBABYSWALL"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlcounterattack(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLCOUNTERATTACK(open,high,low,close)
    col_name="CDLCOUNTERATTACK"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdldarkcloudcover(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLDARKCLOUDCOVER(open,high,low,close)
    col_name="CDLDARKCLOUDCOVER"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdldoji(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLDOJI(open,high,low,close)
    col_name="CDLDOJI"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdldojistar(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLDOJISTAR(open,high,low,close)
    col_name="CDLDOJISTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdldragonflydoji(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLDRAGONFLYDOJI(open,high,low,close)
    col_name="CDLDRAGONFLYDOJI"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlengulfing(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLENGULFING(open,high,low,close)
    col_name="CDLENGULFING"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdleveningdojistar(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLEVENINGDOJISTAR(open,high,low,close)
    col_name="CDLEVENINGDOJISTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdleveningstar(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLEVENINGSTAR(open,high,low,close)
    col_name="CDLEVENINGSTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlgapsidesidewhite(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLGAPSIDESIDEWHITE(open,high,low,close)
    col_name="CDLGAPSIDESIDEWHITE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlgravestonedoji(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLGRAVESTONEDOJI(open,high,low,close)
    col_name="CDLGRAVESTONEDOJI"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlhammer(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHAMMER(open,high,low,close)
    col_name="CDLHAMMER"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlhangingman(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHANGINGMAN(open,high,low,close)
    col_name="CDLHANGINGMAN"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlharami(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHARAMI(open,high,low,close)
    col_name="CDLHARAMI"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlharamicross(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHARAMICROSS(open,high,low,close)
    col_name="CDLHARAMICROSS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlhighwave(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHIGHWAVE(open,high,low,close)
    col_name="CDLHIGHWAVE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlhikkake(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHIKKAKE(open,high,low,close)
    col_name="CDLHIKKAKE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlhikkakemod(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHIKKAKEMOD(open,high,low,close)
    col_name="CDLHIKKAKEMOD"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_cdlhomingpigeon(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLHOMINGPIGEON(open,high,low,close)
    col_name="CDLHOMINGPIGEON"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLIDENTICAL3CROWS(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLIDENTICAL3CROWS(open,high,low,close)
    col_name="CDLIDENTICAL3CROWS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLINNECK(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLINNECK(open,high,low,close)
    col_name="CDLINNECK"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLINVERTEDHAMMER(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLINVERTEDHAMMER(open,high,low,close)
    col_name="CDLINVERTEDHAMMER"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLKICKING(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLKICKING(open,high,low,close)
    col_name="CDLKICKING"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLKICKINGBYLENGTH(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLKICKINGBYLENGTH(open,high,low,close)
    col_name="CDLKICKINGBYLENGTH"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLLADDERBOTTOM(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLLADDERBOTTOM(open,high,low,close)
    col_name="CDLLADDERBOTTOM"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLLONGLEGGEDDOJI(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLLONGLEGGEDDOJI(open,high,low,close)
    col_name="CDLLONGLEGGEDDOJI"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLLONGLINE(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLLONGLINE(open,high,low,close)
    col_name="CDLLONGLINE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLMARUBOZU(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLMARUBOZU(open,high,low,close)
    col_name="CDLMARUBOZU"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLMATCHINGLOW(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLMATCHINGLOW(open,high,low,close)
    col_name="CDLMATCHINGLOW"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLMATHOLD(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLMATHOLD(open,high,low,close)
    col_name="CDLMATHOLD"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLMORNINGDOJISTAR(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLMORNINGDOJISTAR(open,high,low,close)
    col_name="CDLMORNINGDOJISTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLMORNINGSTAR(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLMORNINGSTAR(open,high,low,close)
    col_name="CDLMORNINGSTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLONNECK(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLONNECK(open,high,low,close)
    col_name="CDLONNECK"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLPIERCING(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLPIERCING(open,high,low,close)
    col_name="CDLPIERCING"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLRICKSHAWMAN(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLRICKSHAWMAN(open,high,low,close)
    col_name="CDLRICKSHAWMAN"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLRISEFALL3METHODS(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLRISEFALL3METHODS(open,high,low,close)
    col_name="CDLRISEFALL3METHODS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLSEPARATINGLINES(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLSEPARATINGLINES(open,high,low,close)
    col_name="CDLSEPARATINGLINES"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLSHOOTINGSTAR(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLSHOOTINGSTAR(open,high,low,close)
    col_name="CDLSHOOTINGSTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLSHORTLINE(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLSHORTLINE(open,high,low,close)
    col_name="CDLSHORTLINE"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLSPINNINGTOP(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLSPINNINGTOP(open,high,low,close)
    col_name="CDLSPINNINGTOP"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLSTALLEDPATTERN(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLSTALLEDPATTERN(open,high,low,close)
    col_name="CDLSTALLEDPATTERN"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLSTICKSANDWICH(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLSTICKSANDWICH(open,high,low,close)
    col_name="CDLSTICKSANDWICH"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLTAKURI(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLTAKURI(open,high,low,close)
    col_name="CDLTAKURI"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLTASUKIGAP(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLTASUKIGAP(open,high,low,close)
    col_name="CDLTASUKIGAP"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLTHRUSTING(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLTHRUSTING(open,high,low,close)
    col_name="CDLTHRUSTING"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLTRISTAR(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLTRISTAR(open,high,low,close)
    col_name="CDLTRISTAR"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLUNIQUE3RIVER(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLUNIQUE3RIVER(open,high,low,close)
    col_name="CDLUNIQUE3RIVER"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLUPSIDEGAP2CROWS(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLUPSIDEGAP2CROWS(open,high,low,close)
    col_name="CDLUPSIDEGAP2CROWS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_CDLXSIDEGAP3METHODS(assets):
    df,target_name = ret_df_targetname(assets)
    open=df["open"].values
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    real= ta.CDLXSIDEGAP3METHODS(open,high,low,close)
    col_name="CDLXSIDEGAP3METHODS"
    len_real=len(real)
    target[col_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_beta(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.BETA(high,low, parm[i])
        col_name="BETA"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_correl(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.CORREL(high,low, parm[i])
        col_name="CORREL"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_linearreg(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.LINEARREG(close, parm[i])
        col_name="LINEARREG"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_linearreg_angle(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.LINEARREG_ANGLE(close, parm[i])
        col_name="LINEARREG_ANGLE"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_linearreg_intercept(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.LINEARREG_INTERCEPT(close, parm[i])
        col_name="LINEARREG_INTERCEPT"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_linearreg_slope(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.LINEARREG_SLOPE(close, parm[i])
        col_name="LINEARREG_SLOPE"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_stddev(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.STDDEV(close, parm[i][0],parm[i][1])
        col_name="STDDEV"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_var(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.VAR(close, parm[i][0],parm[i][1])
        col_name="VAR"
        k_name=col_name+"_%s_%s"%(parm[i][0],parm[i][1])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_max(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values


    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.MAX(close, parm[i])
        col_name="MAX"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)

def call_min(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    high=df["high"].values
    low=df["low"].values

    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        real= ta.MIN(close, parm[i])
        col_name="MIN"
        k_name=col_name+"_%s"%(parm[i])
        len_real=len(real)
        target[k_name]=real[len_real-lennn:len_real]
    target.to_csv(target_name, index = None)



def call_roc(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target = pd.DataFrame()
    target["date"]=df["date"]

    #lennn=len(target['Pct_Change_1'].values)
    for i in (parm):
        real=ta.ROC(close,i)
        col_name="Pct_Change_%d"%i
        target[col_name]=(real/100)
    target.to_csv(target_name, index = None)

def call_adj_close(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        col_name="adj_close"
        k_name=col_name+"_%s"%(parm[i])
        target[k_name]=df["close"].shift(abs(parm[i]))
    target.to_csv(target_name, index = None)

def call_volume(assets,parm):
    df,target_name = ret_df_targetname(assets)
    close=df["close"].values
    target=pd.read_csv(target_name)
    lennn=len(target['Pct_Change_1'].values)
    for i in range(len(parm)):
        col_name="volume"
        k_name=col_name+"_%s"%(parm[i])
        target[k_name]=df["volume"].shift(abs(parm[i]))
    target.to_csv(target_name, index = None)








def gen_ta_features(asset):
    
    fram = [1, 2, 5, 8, 10, 19, 20, 50, 80, 100, 150, 180, 200, 250]
    call_roc(asset,fram)
    """    
    zero_parm = parm_list.zero_parm
    one_parm = parm_list.one_parm
    two_parm = parm_list.two_parm
    three_parm = parm_list.three_parm
    four_parm = parm_list.four_parm
    """
    zero_parm = []
    one_parm = [[['rsi'],[5, 10, 14]]]
    two_parm = [
        [['MA'], [[5,0],[10,0],[21,0]]], 
    ]
    three_parm = []
    four_parm = []

    #do 0 parm
    for i in range(len(zero_parm)):
        cmd="call_"+zero_parm[i]+"(asset)"
        #print cmd
        try:
            eval(cmd)
        except:
            print("[{}] failed".format(cmd))
            pass

    #do 1 parm
    for i in range(len(one_parm)):

        table_fmt="one_parm[%s][1]"%(i)
        cmd="call_"+one_parm[i][0][0]+"(asset,"+table_fmt+")"
        #print cmd
        try:
            eval(cmd)
        except:
            print("[{}] failed".format(cmd))
            pass

    #do 2 parm
    for i in range(len(two_parm)):
        # print(two_parm[i][0][0])
        table_fmt="two_parm[%s][1]"%(i)
        cmd="call_"+two_parm[i][0][0]+"(asset,"+table_fmt+")"
        #print cmd
        try:
            eval(cmd)
        except Exception as e:
            print(e)
            print("[{}] failed".format(cmd))
            pass

    #do 3 parm
    for i in range(len(three_parm)):

        table_fmt="three_parm[%s][1]"%(i)
        cmd="call_"+three_parm[i][0][0]+"(asset,"+table_fmt+")"
        #print cmd
        try:
            eval(cmd)
        except:
            print("[{}] failed".format(cmd))
            pass

    #do 4 parm
    for i in range(len(four_parm)):

        table_fmt="four_parm[%s][1]"%(i)
        cmd="call_"+four_parm[i][0][0]+"(asset,"+table_fmt+")"
        #print cmd
        try:
            eval(cmd)
        except:
            print("[{}] failed".format(cmd))
            pass

    return 1






if __name__ == '__main__':
    
    global setting
    setting = build_parser()
    
    
    asset = setting['asset']
    
    price_file = "./data/{}.csv".format(asset)
    
    
    raw_df = get_price(price_file)
    price = raw_df['adj_close']
    
    pps = [1,5,10,21]
    
    for pp in pps:
        df_new = gen_delta_forward(price, pp)
        #print(df_new)
        df_new.to_csv("./data/label_{}_{}.csv".format(asset, pp))
        
        df_new = gen_delta_forward(price, pp, boolean=False)
        #print(df_new)
        df_new.to_csv("./data/pct_change_{}_{}.csv".format(asset, pp))
    
    ret = price.pct_change()

    #span = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377 ,610, 987]
    #span = [1, 2, 3, 5, 8, 13, 21, 34]
    span = range(1,35)
    
    ema_feature = pd.DataFrame()
    for s in span:
        ema = gen_ema_feature(price, s)
        ema_feature = pd.concat([ema_feature, ema], axis=1)
    #print(ema_feature)
    ema_feature.to_csv("./data/EMA_price_{}.csv".format(asset))
    
    
    ema_return = pd.DataFrame()
    for s in span:
        ema = gen_ema_return(price, s)
        ema_return = pd.concat([ema_return, ema], axis=1)
    #print(ema_return)
    ema_return.to_csv("./data/EMA_return_{}.csv".format(asset))
    
    
    cross_feature = gen_cross_feature(span, 2, ema_feature)
    cross_feature_folder = 'features/{}/cross/'.format(asset)
    os.makedirs(cross_feature_folder, exist_ok=True)
    for key, fea in cross_feature.items():
        fea.to_csv(cross_feature_folder+"{}.csv".format(key))
    
    
    extra_feature = gen_extra_feature(raw_df)
    extra_feature_folder = 'features/{}/extra/'.format(asset)
    os.makedirs(extra_feature_folder, exist_ok=True)
    for key, fea in extra_feature.items():
        fea.to_csv(extra_feature_folder+"{}.csv".format(key))

    price_pos = gen_price_position(price)
    price_pos_folder = 'features/price_position/'
    os.makedirs(price_pos_folder, exist_ok=True)
    price_pos.to_csv(price_pos_folder+"price_pos_{}.csv".format(asset))

    extra_feature_folder = 'features/{}/extra/'.format(asset)
    pct_period_start_df = pd.DataFrame()
    for period in ['W', 'M', 'Y']:
        pct_period_start_ser = gen_pct_period_start(price, period)
        pct_period_start_ser.name = '{}_period_start_pct_{}'.format(asset, period)
        pct_period_start_df = pd.concat([pct_period_start_df, pct_period_start_ser], axis=1)
    print(pct_period_start_df)
    pct_period_start_df.to_csv(extra_feature_folder+"pct_period_start_{}.csv".format(asset))

    ma_period_start_df = pd.DataFrame()
    for period in ['W', 'M', 'Y']:
        ma_period_start_ser = gen_ma_period_start(price, period)
        ma_period_start_ser.name = '{}_period_start_ma_{}'.format(asset, period)
        ma_period_start_df = pd.concat([ma_period_start_df, ma_period_start_ser], axis=1)
    print(ma_period_start_df)
    ma_period_start_df.to_csv(extra_feature_folder+"ma_period_start_{}.csv".format(asset))


    global ta_feature_folder
    ta_feature_folder = 'features/{}/allTA/'.format(asset)
    os.makedirs(ta_feature_folder, exist_ok=True)
    gen_ta_features(asset)


    
    
    
    