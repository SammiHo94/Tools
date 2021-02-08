"""API client functions
"""
import logging
import requests
import pandas as pd
import numpy as np
import datetime


def reqTable(url, env, database, table, value_dict={}, select_cols=None):
    """ To get table data from database

    Args:
        url (str): API server end URL with port
        env (str): database machine (e.g. bm1, bm5)
        database (str): name of database (e.g. prod, pipeline)
        table (str): name of table to read (e.g. daily_prediction)
        value_dict (dict, optional): specific column value you want
        select_cols (list, optional): columns that want to show

    Returns:
        pd.DataFrame/pd.Series: table record
    """
    full_url = 'http://{}/get_db/read_table'.format(url)
    params = {
        'env': env,
        'database': database,
        'table': table
    }

    # check type
    for key in value_dict.keys():
        key_value = value_dict[key]
        if isinstance(key_value, list):
            for i, value in enumerate(key_value):
                if not isinstance(value, str):
                    key_value[i] = str(value)
        else:
            if not isinstance(key_value, str):
                key_value = str(key_value)
        value_dict[key] = key_value

    post_dict = {'value_dict': value_dict}
    if select_cols is not None:
        post_dict['select_cols'] = select_cols

    res = requests.get(
        full_url, json=post_dict, params=params)
    html_code = int(res.status_code)
    data = None

    if html_code == 200:
        data_json = res.json()
        logging.info(data_json)
        if 'data' in data_json.keys():
            if isinstance(
                data_json['data'][list(data_json['data'].keys())[0]], dict):
                # for multiple columns result
                data = pd.DataFrame(data_json['data'])
            else:
                # for single column result
                data = pd.Series(data_json['data'])
    elif html_code >= 500:
        msg = 'Server is not responding request'
        logging.debug(msg)
    else:
        logging.debug("Request error code {}".format(res.status_code))

    return data


def reqRebalanceDates(url, env, rebalance_freq,
                      database='prod', table='date_features',
                      sdate=None, edate=None):
    """ To get rebalance date list

    Args:
        url (str): API server end URL with port
        env (str): database machine (e.g. bm1, bm5)
        rebalance_freq (str): rebalance frequence (e.g. 2M, 10D, M+M10)
        database (str, optional): specific database (default prod)
        table (str, optional): specific table to get (default date_features)
        sdate (date/str, optional): specific starting date
        edate (date/str, optional): specific last date of datelist

    Returns:
        pd.Series: rebalance dates list
    """
    full_url = 'http://{}/get_db/rebalance_dates'.format(url)
    params = {
        'env': env,
        'database': database,
        'table': table,
        'rebalance_freq': rebalance_freq,
    }
    if sdate is not None:
        if not isinstance(sdate, str):
            params['sdate'] = str(sdate)
        else:
            params['sdate'] = sdate

    if edate is not None:
        if not isinstance(edate, str):
            params['edate'] = str(edate)
        else:
            params['edate'] = edate

    res = requests.get(full_url, params=params)
    html_code = int(res.status_code)
    datelist = None

    if html_code == 200:
        datelist_json = res.json()
        logging.info(datelist_json)
        if 'data' in datelist_json.keys():
            if isinstance(
                datelist_json['data'][list(datelist_json['data'].keys())[0]], dict):
                # for multiple columns result
                datelist = pd.DataFrame(datelist_json['data'])
            else:
                # for single column result
                datelist = pd.Series(datelist_json['data'])
    elif html_code >= 500:
        msg = 'Server is not responding request'
        logging.debug(msg)
    else:
        logging.debug("Request error code {}".format(res.status_code))
    return datelist


def reqWriteToDB(url, env, database, table, data, if_exists='append'):
    """ Write DataFrame/Series to database

    Args:
        url (str): API server end URL with port
        env (str): database machine (e.g. bm1, bm5)
        database (str): name of database (e.g. prod, pipeline)
        table (str): name of table to read (e.g. daily_prediction)
        data (pd.DataFrame/pd.Series): data to write,
            process will ignore index, please reset index before call this function

    Returns:
        str: response message, show success or fail
    """
    full_url = 'http://{}/post2db/write2db'.format(url)
    if isinstance(data, pd.Series):
        data = data.to_frame().reset_index()

    # make sure format won't change after send to server
    for col in data.columns:
        if isinstance(data[col][0], (np.datetime64, datetime.date, datetime.datetime)):
            data[col] = data[col].astype(str)
    post_dict = data.to_dict(orient='list')

    params = {
        'env': env,
        'database': database,
        'table': table,
        'col_order': list(data.columns),
        'if_exists': if_exists
    }

    res = requests.post(full_url, json=post_dict, params=params)
    html_code = int(res.status_code)
    res_json = None

    if html_code == 200:
        res_json = res.json()
        logging.info(res_json)
    elif html_code >= 500:
        msg = 'Server is not responding request'
        logging.debug(msg)
    else:
        logging.debug("Request error code {}".format(res.status_code))

    if res_json is not None and res_json['ok']:
        msg = 'Success to post'
    else:
        msg = "Failed to post"
    return msg


def reqUpdateTable(url, env, database, table, value_dict, update_dict):
    """ Write DataFrame/Series to database

    Args:
        url (str): API server end URL with port
        env (str): database machine (e.g. bm1, bm5)
        database (str): name of database (e.g. prod, pipeline)
        table (str): name of table to read (e.g. daily_prediction)
        value_dict (dict): identify which rows will be updated
        update_dict (dict): update values

    Returns:
        str: response message, show success or fail
    """
    full_url = 'http://{}/post2db/update_table'.format(url)

    params = {
        'env': env,
        'database': database,
        'table': table
    }
    post_dict = {
        'value_dict': value_dict,
        'update_dict': update_dict
    }

    res = requests.post(full_url, json=post_dict, params=params)
    html_code = int(res.status_code)
    res_json = None

    if html_code == 200:
        res_json = res.json()
        logging.info(res_json)
    elif html_code >= 500:
        msg = 'Server is not responding request'
        logging.debug(msg)
    else:
        logging.debug("Request error code {}".format(res.status_code))

    if res_json is not None and res_json['ok']:
        msg = 'Success to post'
    else:
        msg = "Failed to post"
    return msg


def reqSendEmail(url, emsubject, receiver, email_text):
    """ send email by alert.automatedml@gmail.com

    Args:
        url (str): API server end URL with port
        emsubject (str): email subject
        receiver (list): list of email receiver
        email_text (str): email contents

    Returns:
        str: response message, show success or fail
    """
    full_url = 'http://{}/send_email'.format(url)
    params = {
        'emsubject': emsubject,
        'receiver': receiver,
        'email_text': email_text
    }
    res = requests.post(full_url, params=params)
    html_code = int(res.status_code)
    res_json = None

    if html_code == 200:
        res_json = res.json()
        logging.info(res_json)
    elif html_code >= 500:
        msg = 'Server is not responding request'
        logging.debug(msg)
    else:
        logging.debug("Request error code {}".format(res.status_code))

    if res_json is not None and res_json['ok']:
        msg = 'Success to send email'
    else:
        msg = "Failed to send email"
    return msg
