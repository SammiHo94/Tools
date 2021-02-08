""" API functions handler
"""
import datetime
import pandas as pd
import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from classes.email import MailSender
from classes.gendatelist import DateListGenerator


def read_table(server, database, table, **kwargs):
    """ To get table data from database

    Args:
        server (object): database connection
        database (str): name of database (e.g. prod, pipeline)
        table (str): name of table to read (e.g. daily_prediction)
        **kwargs: other optional parameters

    Returns:
        dict: status and result
    """
    res = {'ok': False}

    try:
        if 'value_dict' in kwargs:
            value_dict = kwargs['value_dict']
        else:
            value_dict = {}

        if 'select_cols' in kwargs:
            select_cols = kwargs['select_cols']
        else:
            select_cols = None

        tb_obj, stmt = server.query_table(
            database, table, select_cols, value_dict)
        df = pd.read_sql(stmt, server.engine)

        if not df.empty:
            # make sure format won't change after send to client
            for col in df.columns:
                if isinstance(df[col][0], (np.datetime64, datetime.date, datetime.datetime)):
                    df[col] = df[col].astype(str)

            if len(df.index) == 1:
                res['data'] = df.to_dict()
            else:
                res['data'] = df.squeeze().to_dict()
            res['ok'] = True
        else:
            res['error'] = 'No record'
    except SQLAlchemyError:
        res['error'] = 'SQL'

    server.close()

    return res


def read_rebalance_dates(server, database, table, rebalance_freq, **kwargs):
    """ To get rebalance date list

    Args:
        server (object): database connection
        database (str): name of database (e.g. prod)
        table (str): name of table to read (e.g. date_features)
        rebalance_freq (str): rebalance frequence (e.g. 2M, 10D, M+M10)
        **kwargs: other optional parameters

    Returns:
        dict: status and result
    """
    res = {'ok': False}

    try:
        tb_obj, stmt = server.query_table(database, table)
        df = pd.read_sql(stmt, server.engine)
        if not df.empty:
            datelist_generator = DateListGenerator(df)
            rebalance_freq = rebalance_freq[0]

            if 'sdate' in kwargs:
                sdate = datetime.datetime.strptime(kwargs['sdate'][0], '%Y-%m-%d').date()
            else:
                sdate = None

            if 'edate' in kwargs:
                edate = datetime.datetime.strptime(kwargs['edate'][0], '%Y-%m-%d').date()
            else:
                edate = datetime.date.today()

            datelist = datelist_generator.gen_datelist(
                rebalance_freq, sdate, edate)
            datelist = datelist.astype('str')

            if len(datelist.index) == 1:
                res['data'] = datelist.to_dict()
            else:
                res['data'] = datelist.squeeze().to_dict()
            res['ok'] = True
        else:
            res['error'] = 'No date features record'
    except SQLAlchemyError:
        res['error'] = 'SQL'
    server.close()
    return res


def send_email(emsubject, receiver, email_text, **kwargs):
    """ send email by alert.automatedml@gmail.com

    Args:
        emsubject (str): email subject
        receiver (list): list of email receiver
        email_text (str): email contents
        **kwargs: other optional parameters

    Returns:
        dict: status and result
    """
    res = {'ok': False}
    try:
        aml_email = MailSender()
        aml_email.send(receiver, emsubject[0], email_text[0])
    except:
        res['error'] = 'sendgrid error'
    else:
        res['ok'] = True
    return res


def writeDf2DB(server, database, table, data, **kwargs):
    """ Write data dict to database

    Args:
        server (object): database connection
        database (str): name of database (e.g. prod, pipeline)
        table (str): name of table to read (e.g. daily_prediction)
        data (dict): data to write
        **kwargs: other optional parameters

    Returns:
        dict: status and result
    """
    res = {'ok': False}
    df = pd.DataFrame.from_dict(data, orient='columns')

    if isinstance(database, list):
        database = database[0]

    if isinstance(table, list):
        table = table[0]

    if 'col_order' in kwargs:
        cols = kwargs['col_order']
        df = df[cols]

    if 'if_exists' in kwargs:
        if_exists = kwargs['if_exists'][0]
    else:
        if_exists = 'append'

    try:
        server.select_database(database)
        df.to_sql(table, server.engine, if_exists=if_exists, index=False)
        res['ok'] = True
    except SQLAlchemyError:
        res['error'] = 'SQL'
    server.close()
    return res


def updateDBtable(server, database, table, data, **kwargs):
    """ Write data dict to database

    Args:
        server (object): database connection
        database (str): name of database (e.g. prod, pipeline)
        table (str): name of table to read (e.g. daily_prediction)
        data (dict): value_dict (dict): identify which rows will be updated
                     and update_dict (dict): update values
        **kwargs: other optional parameters

    Returns:
        dict: status and result
    """
    res = {'ok': False}

    value_dict = data['value_dict']
    update_dict = data['update_dict']

    if isinstance(database, list):
        database = database[0]

    if isinstance(table, list):
        table = table[0]

    try:
        server.update_table(database, table, value_dict, update_dict)
        res['ok'] = True
    except SQLAlchemyError:
        res['error'] = 'SQL'
    server.close()
    return res
