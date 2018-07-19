"""
This package provides the format function.
"""
import json


def format_json(obj):
    """
    format json string
    :param obj:
    :return:
    """
    result = json.dumps(obj, indent=4, ensure_ascii=False, sort_keys=True)
    return result


def table_print(tuple_list):
    """
    output data in the table format
    :param tuple_list
    :return:
    """
    sub1 = [
        [str(s).ljust(max(len(str(i)) for i in grp)) for s in grp]
        for grp in zip(*tuple_list)]
    header = True
    for p in [" | ".join(row) for row in zip(*sub1)]:
        if header:
            print len(p)*"="
            print p
            # print("\033[1;31m" + p + "\033[0m")
            print len(p)*"-"
            header = False
        else:
            print p


def pretty_print(obj):
    """
    output data in the vertical format
    :param obj: dict type
    :return:
    """
    num = 0
    if len(obj) < 1:
        print "No data."
        exit(0)
    for i in obj:
        num += 1
        keys = tuple(i.keys())
        values = tuple(i.values())
        sub1 = [
            [str(s).rjust(max(len(str(i)) for i in keys)) for s in keys],
            [str(s).ljust(max(len(str(i)) for i in values)) for s in values]]
        print "*************************** %s. row ***************************" % num
        for p in [" : ".join(row) for row in zip(*sub1)]:
            print p
