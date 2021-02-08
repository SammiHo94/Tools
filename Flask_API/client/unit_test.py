import unittest

import api_client as api
import pandas as pd


url = '192.168.33.107:15003'

class TestAPI(unittest.TestCase):

    def test_read_table(self):
        test_table = api.reqTable(
            url, 'bm5', 'prod', 'date_features', value_dict={'date': '2020-07-27'})
        print('test_table', test_table)
        self.assertIsInstance(
            test_table, pd.DataFrame, "result is not dataframe")

    def test_read_dates(self):
        test_table = api.reqTable(
            url, 'bm5', 'prod', 'date_features', value_dict={'date': ['2012-07-01', '2012-12-30']})
        print('test_table', test_table)
        self.assertIsInstance(
            test_table, pd.DataFrame, "result is not dataframe")

    def test_read_table(self):
        test_table = api.reqTable(url, 'bm1', 'pipeline', 'mpt_live')
        self.assertIsInstance(
            test_table, pd.DataFrame, "result is not dataframe")

    def test_read_table_with_date(self):
        test_table_date = api.reqTable(
            url, 'bm1', 'pipeline', 'mpt_live', {'date': '2019-05-01'})
        print(test_table_date)
        self.assertIsInstance(
            test_table_date, pd.DataFrame, "result is not dataframe")

    def test_read_table_with_cols(self):
        test_table_col = api.reqTable(
            url, 'bm5', 'prediction_test', 'daily_prediction',
            {'Trading Date': '2020-08-03', 'Symbol': ['DBC', 'IYR']},
            ['Execution Time', 'Symbol', 'Trading Date', 'Trend'])
        print(test_table_col)
        self.assertIsInstance(
            test_table_col, pd.DataFrame, "result is not dataframe")

    def test_send_email(self):
        testemail = api.reqSendEmail(
            url, 'email_subject', ['sammiho.aml@gmail.com'], 'test_email api')
        self.assertNotIn('Failed', testemail, 'Failed')

    def test_rabalance_date(self):
        test_rebalabnce = api.reqRebalanceDates(url, 'bm1', 'M')
        self.assertIsInstance(
            test_rebalabnce, pd.Series, "result is not Series")
        self.assertFalse(test_rebalabnce.empty, 'result is empty')

    def test_write2db(self):
        df = pd.DataFrame(index=range(0, 5))
        df['colA'] = ['A', 'B', 'C', 'D', 'E']
        df['colB'] = ['a', 'b', 'c', 'd', 'e']
        test_writedf2DB = api.reqWriteToDB(
            url, 'bm1', 'pipeline', 'test_api', df)
        self.assertNotIn('Failed', test_writedf2DB, 'Failed')

    def test_update(self):
        from datetime import datetime
        value_dict = {
            'exec_date': '2020-03-16',
            'asset': "DBC",
            'data_mode': 'combine_all',
            'model_group': 'sk',
            'process': 'train_models'
        }
        update_dict = {
            'start_time': datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        }

        test_update = api.reqUpdateTable(
            url, 'bm5', 'job_status', 'beta_jobs_test', value_dict, update_dict)
        self.assertNotIn('Failed', test_update, 'Failed')


if __name__ == '__main__':
    unittest.main()
