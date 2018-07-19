"""
This package provides the cluster commands.
"""
import sys
import paddlecli.conf.api as _api
import paddlecli.core.http.http_client as http_client
import paddlecli.core.http.http_method as http_method
from paddlecli.lib.format import table_print
from paddlecli.lib.format import format_json

import json


def items(args):
    """
    list cluster name, cluster type etc. list
    """
    cluster_list = get_cluster_list(args)
    if len(cluster_list) == 0:
        print "Oops...There are no available clusters in the current token."
        sys.exit(1)
    tuple_list = make_tuple_list_with_keys(cluster_list)
    table_print(tuple_list)


def info(args):
    """
    get cluster config info
    """
    params = dict()
    params['cluster_name'] = args.cluster_name
    response = http_client.send_request(
        http_method.GET,
        _api.QUOTA_URL,
        None, None, params, None, args)
    obj = json.loads(response.read())
    if obj['code'] != 0:
        raise Exception(obj['msg'])
    print format_json(obj)


def make_tuple_list_with_keys(obj):
    """
    make tuple list with keys
    """
    tuple_list = []
    flag = True
    for i in obj:
        if flag:
            keys = tuple(i.keys())
            tuple_list.append(keys)
        flag = False
        item = tuple(i.values())
        if item not in tuple_list:
            tuple_list.append(item)
    return tuple_list


def get_cluster_names(cluster_list):
    """
    get cluster names
    """
    name_list = set()
    for i in cluster_list:
        name_list.add(i['cluster_name'])
    return name_list


def get_cluster_list(args):
    """
    get cluster list
    """
    params = dict()
    response = http_client.send_request(
        http_method.GET,
        _api.USERS_AUTH_URL,
        None, None, params, None, args)
    content = response.read()
    json_obj = json.loads(content)
    if json_obj['code'] != 0:
        raise Exception(json_obj['msg'])
    cluster_list = []
    for data in json_obj['data']:
        phy_list = data['phy']
        for phy in phy_list:
            cluster = phy['cluster']
            # cluster_name = cluster['name']
            # cluster['group_name'] = data['name']
            cluster['cluster_name'] = cluster['name']
            del cluster['name']
            if 'host_name' in cluster:
                del cluster['host_name']
            if cluster not in cluster_list:
                cluster_list.append(cluster)

    return cluster_list
