import os
import traceback
import datetime

from configparser import ConfigParser
from flask import Flask, jsonify
from flask import request as flask_req
from flask import abort

from classes.mysql import database

debug = True


def read_dbconf(path=None):
    if path is None:
        path = './config/mysql.ini'
    dbcfg = ConfigParser()
    dbcfg.read(path)

    for key in dbcfg._sections:
        dbcfg[key]['password'] = os.environ.get('%s_db_pw' % key)
    print(dbcfg)
    return dbcfg


def create_app():
    global app
    app = Flask(__name__)

    def loc_log(url):
        os.makedirs(log_dir, exist_ok=True)
        sec = 'API-{}'.format(url)
        if sec in func_cfg.keys():
            file_name = func_cfg[sec]['logfile']
        else:
            file_name = 'main'
        return file_name

    @app.before_request
    def record_request():
        now = datetime.datetime.now()
        file_name = loc_log(flask_req.path)
        file = open(log_dir + file_name + log_ext, 'a')
        file.write('Request at [{}]\n'.format(now))
        args = flask_req.args.to_dict(flat=False).items()
        form = flask_req.form.to_dict().items()
        for x, (i, j) in enumerate(args, start=1):
            if x == 1:
                file.write('with Arguments:\n')
            file.write('{} {} -> {}\n'.format(x, i, j))
        for x, (i, j) in enumerate(form, start=1):
            if x == 1:
                file.write('with Form:\n')
            file.write('{} {} -> {}\n'.format(x, i, j))
        json = flask_req.get_json(silent=True)
        if json is not None:
            for x, (i, j) in enumerate(json.items(), start=1):
                if x == 1:
                    file.write('with JSON:\n')
                file.write('{} {} -> {}\n'.format(x, i, j))
        file.write('--- End of Request ---\n')
        file.close()

    @app.errorhandler(Exception)
    def record_error(error):
        now = datetime.datetime.now()
        file_name = loc_log(flask_req.path)
        file = open(log_dir + file_name + log_ext, 'a')
        file.write('Error:\n')
        file.write('{} {} {} {} {} \n{}\n'.format(
            now,
            flask_req.remote_addr,
            flask_req.method,
            flask_req.scheme,
            flask_req.full_path,
            traceback.format_exc()))
        file.close()
        return "Internal Server Error", 500

    @app.after_request
    def record_response(res):
        now = datetime.datetime.now()
        file_name = loc_log(flask_req.path)
        file = open(log_dir + file_name + log_ext, 'a')
        if res.status_code != 500:
            file.write('Response:\n')
            file.write('{} {} {} {} {} {}\n,'.format(
                now,
                flask_req.remote_addr,
                flask_req.method,
                flask_req.scheme,
                flask_req.full_path,
                res.status))
            file.write('{}'.format(res.data.decode('utf-8')))
            file.write('--- End of Response ---\n')
            file.close()
        return res


def lowercase_key(my_dict):
    if not isinstance(my_dict, dict):
        raise TypeError('Input Variable Type: dictionary')
    return {k.lower(): v for k, v in my_dict.items()}


def api_from_cfg():
    api_cfg = {}
    for i, j in func_cfg.items():
        if i.startswith('API-'):
            key_name = i[4:]
            api_cfg[key_name] = dict(j)
            file, var = api_cfg[key_name]['func_path'].rsplit('.', 1)
            exec('from classes.{} import {}'.format(file, var))
            api_cfg[key_name]['func'] = eval(var)
    return api_cfg


def check_authorization():
    """
    TBC
    """
    pass


def url_routing():
    @app.route('/', methods=['get'])
    def home():
        return 'Welcome to AML API'

    @app.route('/get_db/<string:target>', methods=['get'])
    def get_db_data(target):
        args = flask_req.args.to_dict(flat=False)
        json = flask_req.get_json(silent=True)
        if json is None:
            json = {}
        if flask_req.path not in api.keys():
            abort(403)
        this_api = api[flask_req.path]

        env = args['env'][0]
        if 'table' in args and not isinstance(args['table'], str):
            args['table'] = args['table'][0]

        if 'database' in args and not isinstance(args['database'], str):
            args['database'] = args['database'][0]

        if env not in dbcfg:
            results = {'ok': False, 'error': "No can't find env in DB config"}
        else:
            this_api['server'] = database(**dbcfg[env])

            all_paras = lowercase_key(dict(json, **this_api, **args))
            results = this_api['func'](**all_paras)
        return jsonify(results)

    @app.route('/post2db/<string:target>', methods=['post'])
    def post2db(target):
        args = flask_req.args.to_dict(flat=False)
        json = flask_req.get_json(silent=True)
        if json is None:
            json = {}
        if flask_req.path not in api.keys():
            abort(403)

        this_api = api[flask_req.path]

        env = args['env'][0]
        if env not in dbcfg:
            results = {'ok': False, 'error': "No can't find env in config"}
        else:
            this_api['server'] = database(**dbcfg[env])

            all_paras = lowercase_key(dict(**this_api, **args))
            all_paras['data'] = json
            results = this_api['func'](**all_paras)
        return jsonify(results)

    @app.route('/send_email', methods=['post'])
    def reqSendMail():
        args = flask_req.args.to_dict(flat=False)
        json = flask_req.get_json(silent=True)
        if json is None:
            json = {}
        if flask_req.path not in api.keys():
            abort(403)
        this_api = api[flask_req.path]

        all_paras = lowercase_key(dict(**this_api, **args))
        all_paras['data'] = json
        results = this_api['func'](**all_paras)
        return jsonify(results)


if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read("./config/api.ini")

    log_dir = cfg['API']['log_dir']
    log_ext = cfg['API']['log_ext']
    os.makedirs(log_dir, exist_ok=True)

    db_map = cfg['API']['db_config']
    dbcfg = read_dbconf(db_map)

    func_map = cfg['API']['func_config']
    func_cfg = ConfigParser()
    func_cfg.read(func_map)

    api = api_from_cfg()
    create_app()
    url_routing()
    app.run(host='0.0.0.0', port=80)
