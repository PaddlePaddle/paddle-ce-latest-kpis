"""
This module provide customized ArgumentParser.
"""
import paddlecli
import paddlecli.conf.user
import paddlecli.conf.server
from argparse import ArgumentParser
from gettext import gettext as _
from paddlecli.argparser.job import build_job_parser
from paddlecli.argparser.cluster import build_cluster_parser
from paddlecli.argparser.token import build_token_parser

VERSION_MSG = "paddle cloud platform v" + paddlecli.CLI_VERSION


class PaddleCloudParser(ArgumentParser):
    """
    Specialized Argument Parser, print help message instead of usage when error.
    """
    def _check_value(self, action, value):
        # converted value must be one of the choices (if specified)
        if action.choices is not None and value not in action.choices:
            tup = "Oops,", value, self.prog
            msg = _("%s '%s' is not a valid command or arg. See '%s --help'") % tup
            self.exit(2, msg)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        import sys

        print _('%s: error: %s') % (self.prog, message)
        self.print_help(sys.stderr)
        self.exit(2)


def build_argument_parser():
    """
    build paddlecloud client argument parser
    :return:
    """
    parser = PaddleCloudParser()
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=VERSION_MSG)

    sub_parsers = parser.add_subparsers(
        dest='srv',
        title='available sub commands',
        metavar='<command> [option]'
    )

    token_parser = sub_parsers.add_parser(
        'token',
        help='add, remove, switch and list tokens'
    )

    job_parser = sub_parsers.add_parser(
        'job',
        help='train, list, kill or delete jobs',
    )

    config_parser = sub_parsers.add_parser(
        'config',
        help='configure access token, server or port',
    )
    config_parser.set_defaults(func=paddlecli.conf.user.help)
    config_parser.add_argument(
        '-t',
        '--token',
        nargs=0,
        help="configure user's access key and secret key",
        action=paddlecli.conf.user.UserConfigAction,
        metavar=''
    )
    config_parser.add_argument(
        '-s',
        '--server',
        nargs=0,
        help='configure paddlecloud server host, port',
        action=paddlecli.conf.server.ServerConfigAction,
        metavar=''
    )

    cluster_parser = sub_parsers.add_parser(
        'cluster',
        help='list or show cluster information'
    )

    config_parser_dict = config_parser
    job_parser_dict = build_job_parser(job_parser)
    cluster_parser_dict = build_cluster_parser(cluster_parser)
    token_parser_dict = build_token_parser(token_parser)

    parser_dict = {
        'root': parser,
        'job': job_parser_dict,
        'cluster': cluster_parser_dict,
        'config': config_parser_dict,
        'token': token_parser_dict,
    }
    return parser_dict
