"""
This module provide the sub-arg parser for cluster.
"""

import paddlecli
import paddlecli.core.cluster


def __build_list_parser(parser):
    """
    list cluster parser
    """
    parser.set_defaults(func=paddlecli.core.cluster.items)


def __build_info_parser(parser):
    """
    cluster info parser
    """
    parser.set_defaults(func=paddlecli.core.cluster.info)
    required_parser = parser.add_argument_group('required  arguments')
    required_parser.add_argument(
        '--cluster-name',
        required=True,
        help='set cluster name',
        metavar='<ARG>',
        dest='cluster_name'
    )


def build_cluster_parser(parser):
    """
    cluster parser builder
    """

    cluster_parser_dict = dict()
    cluster_parser_dict["root"] = parser

    parser.add_argument(
        '--server',
        type=str,
        action='store',
        help='specify remote server host',
        metavar='<ARG>',
        dest='server'
    )
    parser.add_argument(
        '--port',
        action='store',
        type=int,
        help='specify remote server port',
        metavar='<ARG>',
        dest='port'
    )
    parser.add_argument(
        '--user-ak',
        action='store',
        type=str,
        help='specify user access key',
        metavar='<ARG>',
        dest='user_ak'
    )
    parser.add_argument(
        '--user-sk',
        action='store',
        type=str,
        help='specify user secret key',
        metavar='<ARG>',
        dest='user_sk'
    )

    sub_parsers = parser.add_subparsers(
        dest='cmd',
        title='Specific cluster actions',
        metavar='<action> [options]'
    )

    list_parser = sub_parsers.add_parser(
        'list',
        help='list available cluster items'
    )
    cluster_parser_dict["list"] = list_parser

    info_parser = sub_parsers.add_parser(
        'info',
        help='show the cluster resource information'
    )
    cluster_parser_dict["info"] = info_parser

    __build_list_parser(list_parser)
    __build_info_parser(info_parser)

    return cluster_parser_dict
