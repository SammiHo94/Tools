"""get datelist from date-feature table in database

Attributes:
    E_DATE (date): today
"""
import pandas as pd
import re
from datetime import date

E_DATE = date.today()


class DateListGenerator():

    """
    Attributes:
        connect : Database connection
    """

    def __init__(self, df):
        self.all_tradedates = df

    def get_tradedate(self):
        """get trade date series from database

        Args:
            table (str): date features table name
        Returns:
            Series: All trade dates
        """

        dfX = self.all_tradedates
        tradedates = dfX['date']
        tradedates = pd.to_datetime(tradedates, format='%Y-%m-%d').dt.date
        return tradedates

    def check_tradedate(self, this_date, rebalance_freq):
        """check whether this_date is trade date

        Args:
            this_date (date): the date to check
            rebalance_freq (str): how long will rebalance

        Returns:
            int: day of month for that date
        """

        dfX = self.all_tradedates
        tradedates = dfX.set_index('date')
        tradedates.index = pd.to_datetime(tradedates.index, format='%Y-%m-%d').date

        if this_date not in tradedates.index:
            return -1

        else:
            datelist_all = pd.DataFrame(list(tradedates.index), columns=['date'])
            datelist_all.index = datelist_all.date

            datelist_rb = self.rebalance_date(datelist_all.index, rebalance_freq)
            datelist_rb[datelist_rb <= this_date]

            if this_date == datelist_rb.iloc[-1]:
                return 1
            else:
                return 2

    def gen_datelist(self, rebalance_freq, begin_date=None,
                     end_date=E_DATE):
        """Gen predict date list based on MPT rule

        Args:
            rebalance_freq (str): how long will rebalance
            maxdate (date): last data's date
            begin_date (None, optional): First data date should later than this date
            end_date (TYPE, optional): Last data date should earier than this date

        Returns:
            pd.DataFrame: list all prediction date
        """
        dfX = self.all_tradedates

        datelist = pd.to_datetime(dfX['date'], format='%Y-%m-%d').dt.date
        datelist_all = pd.DataFrame(list(datelist), columns=['date'])
        datelist_all.index = datelist_all.date

        # cut begin-date if set
        if begin_date is not None:
            datelist_all = datelist_all[datelist_all.index >= begin_date]

        # cut end-date if set
        if end_date is not None:
            datelist_all = datelist_all[datelist_all.index <= end_date]

        datelist_rb = self.rebalance_date(datelist_all.index, rebalance_freq)
        datelist_rb = datelist_rb.reset_index(drop=True)

        return datelist_rb

    def rebalance_date(self, dates, freq='M'):
        """Get rebalance date base on frequence

        Args:
            dates (Index): All trade dates in special period
            freq (str, optional): the rebalance frequence

        Returns:
            pd.DataFrame: the selected rebalance date
        """
        dates = pd.to_datetime(dates)
        dts = pd.Series(dates, index=dates).sort_values()
        if '+' in freq:
            temp = []
            for sub_freq in freq.split('+'):
                sub_dts = self.group_series(dts, *self.parse_freq(sub_freq))
                temp.append(sub_dts)
            results = pd.concat(temp).drop_duplicates().sort_values()
        else:
            results = self.group_series(dts, *self.parse_freq(freq))

        results = results.reset_index(drop=True).rename(freq)
        results_df = results.to_frame()
        results_df[freq] = pd.to_datetime(
            results_df[freq], format='%Y-%m-%d').dt.date

        return results_df[freq]

    @staticmethod
    def group_series(dates, n_period, period, nth):
        """
        Args:
            dates (pd.Series): All trade dates in special period
            n_period (int): number of period
            period (str): frequency period
            nth (int): number of start point

        Returns:
            pd.DataFrame: the selected rebalance date
        """
        err = 'only support pandas.datetimeindex'
        assert isinstance(dates.index, pd.DatetimeIndex), err
        period_index = dates.index.to_period(period)
        if period != 'D':
            results = dates.groupby(period_index).nth(nth)
        else:
            results = dates[nth::n_period]
            n_period = 0
        if n_period > 1:
            results = results[::int(n_period)]
        return results.reset_index(drop=True)

    @staticmethod
    def parse_freq(s, default_num=1, default_nth=0):
        """Split frequency
        Args:
            s (str): rebalance frequence (e.g. 2M, 10D)
            default_num (int, optional): number of period
            default_nth (int, optional): number of start point

        Returns:
            num (int): number of period
            p (str): frequency period
            nth (int): number of start point

        Raises:
            KeyError: Wrong input format of Frequency
        """
        r = re.compile('(|\d+)([A-Za-z]+)(\d+|)')
        match = r.match(s)
        if not match:
            raise KeyError('Invalid format of Frequency [{}]'.format(s))
        num, p, nth = match.groups()
        num = int(num) if len(num) > 0 else default_num
        nth = int(nth) if len(nth) > 0 else default_nth
        return num, p, nth
