"""
This module provide the sub-arg parser for cluster.
"""

import paddlecli
import paddlecli.conf.token


def build_token_parser(parser):
    """
    cluster parser builder
    """

    token_parser_dict = dict()
    token_parser_dict["root"] = parser
    parser.set_defaults(func=paddlecli.conf.token.current)
    parser.add_argument(
        '-a',
        '--add',
        nargs=0,
        help='add a token in local',
        action=paddlecli.conf.token.AddToken,
        metavar=''
    )

    parser.add_argument(
        '-l',
        '--list',
        nargs=0,
        help='list all tokens',
        action=paddlecli.conf.token.QueryAllTokens,
        metavar=''
    )
    parser.add_argument(
        '--switch',
        nargs=1,
        help='switch to another token',
        action=paddlecli.conf.token.SwitchToken,
        metavar='<ARG>'
    )
    parser.add_argument(
        '--remove',
        nargs=1,
        help='remove an existing token',
        action=paddlecli.conf.token.RemoveToken,
        metavar='<ARG>'
    )

    return token_parser_dict
