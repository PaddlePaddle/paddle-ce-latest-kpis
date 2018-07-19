#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This package provides the paddlecloud command-line tool.
"""

import sys
import codecs
from paddlecli.argparser.parser import build_argument_parser


def main():
    """
    the command-line tool entrance
    :return:
    """
    # Unique every input string to unicode encoding string
    def _unique_encoding(x):
        input_encoding = sys.stdin.encoding
        if input_encoding is not None:
            return x.decode(input_encoding)
        return x
    sys.argv = map(_unique_encoding, sys.argv)
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr)
    try:
        parser_dict = build_argument_parser()
        parser = parser_dict['root']
        args, unknown = parser.parse_known_args()
        if len(unknown) != 0:
            print "error: unknown option '%s'" % (" ".join(unknown))
            if hasattr(args, 'cmd'):
                parser_dict[args.srv][args.cmd].print_help()
            else:
                parser_dict[args.srv].print_help()

            sys.exit(1)
        else:
            if hasattr(args, 'func'):
                args.func(args)
    except Exception as e:
        sys.stderr.write("Error occurred: %s\n" % e)
        sys.exit(1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
