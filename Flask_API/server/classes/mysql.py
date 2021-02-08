from sqlalchemy import MetaData, create_engine


class database(object):

    def __init__(self, dialect, driver,
                 username, password,
                 host, port=3306, db=''):
        self.base_url = join_db_url(dialect, driver,
                                    username, password, host, port)
        self.select_database(db)

    def select_database(self, db):
        if hasattr(self, 'cdb'):
            if self.cdb == db:
                return self.engine
        url = self.base_url + db
        self.close()
        self.engine = create_engine(url)
        self.connect()
        self.cdb = db
        self.trans = self.connection.begin()
        if db is not '':
            self.metadata = MetaData()
            self.metadata.reflect(self.connection)
            self.tables = self.metadata.tables
        return

    def connect(self):
        self.connection = self.engine.connect()
        return self.engine

    def close(self):
        if hasattr(self, 'connection'):
            self.connection.close()

    def select_table(self, db, tb_name):
        self.select_database(db)
        if tb_name not in self.tables.keys():
            raise KeyError('Table [{}] not found'.format(tb_name))
        return self.tables[tb_name]

    def query_table(self, db, tb_name, select_cols=None, value_dict={}):
        """ query table

        Args:
            db (str): Name of database
            tb_name (str): Name of tables
            value_dict (dict, optional): specific column value you want
            select_cols (list, optional): columns that want to show

        Returns:
            obj: table object
            str: query sentence
        """
        from sqlalchemy.sql.expression import select
        tb = self.select_table(db, tb_name)
        if select_cols is None:
            qry = select([tb.c[i] for i in tb.c.keys()])
        else:
            for col in value_dict.keys():
                if col not in select_cols:
                    select_cols.append(col)
            qry = select([tb.c[i] for i in select_cols if i in tb.c.keys()])

        for key_col in value_dict.keys():
            if key_col in tb.c.keys():
                my_col = tb.c[key_col]
                value = value_dict[key_col]
                if key_col == 'date' and isinstance(value, list):
                    qry = qry.where((my_col >= str(value[0])) & (
                        my_col < str(value[1])))
                elif isinstance(value, list):
                    qry = qry.where(my_col.in_(value))
                else:
                    qry = qry.where(my_col == str(value))
        return tb, qry

    def update_table(self, db, tb_name, value_dict, update_dict):
        """ update table values

        Args:
            db (str): Name of database
            tb_name (str): Name of tables
            value_dict (dict): identify which rows will be updated
            update_dict (dict): update values
        """
        from sqlalchemy.sql.expression import update
        tb = self.select_table(db, tb_name)
        query = update(tb)

        # identify which rows will be updated
        for idx_col in value_dict.keys():
            if idx_col in tb.c.keys():
                query = query.where(tb.c[idx_col] == str(value_dict[idx_col]))

        # match update column and values
        query = query.values(update_dict)

        # execute update sentence
        self.engine.execute(query)
        self.close()


def join_db_url(dialect, driver, username, password, host, port):
    # url = dialect+driver://username:password@host:port/database
    url = '{}+{}://'.format(dialect, driver)
    url += '{}:{}'.format(username, password)
    url += '@{}:{}/'.format(host, port)
    return url
