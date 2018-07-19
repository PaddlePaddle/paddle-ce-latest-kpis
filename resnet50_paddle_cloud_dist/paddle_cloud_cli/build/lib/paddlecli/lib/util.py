"""
This package provides the common function.
"""

import datetime
import hashlib
import hmac
import time
import string
import uuid
import re
import paddlecli
from paddlecli.core.http import http_headers

PADDLE_PREFIX = "x-paddle-"


def _get_normalized_char_list():
    """
    get normalized char list
    :return:
    """
    ret = ['%%%02X' % i for i in range(256)]
    for ch in string.ascii_letters + string.digits + '.~-_':
        ret[ord(ch)] = ch
    return ret


_NORMALIZED_CHAR_LIST = _get_normalized_char_list()


def get_canonical_uri(path):
    """
    generate canonical URI
    """
    # canonical uri demo: /{bucket}/{object}, and encode all characters except the slash "/"
    return normalize_string(path, False)


def get_canonical_time(timestamp=0):
    """
    Get canonical time.
    :type timestamp: int
    :param timestamp: None
    =======================
    :return:
        **string of canonical_time**
    """
    if timestamp == 0:
        timestamp = time.time()
    utctime = datetime.datetime.fromtimestamp(timestamp)
    return "%04d-%02d-%02dT%02d:%02d:%02dZ" % (
        utctime.year, utctime.month, utctime.day,
        utctime.hour, utctime.minute, utctime.second)


def sign_func(config, http_method, path, headers):
    """
    compute auth info using ak sk method path date expires
    sing = md5.hexdigest()
    """
    md5 = hashlib.md5()
    all = "%s%s%s%s%s%s" % (config.user_ak, config.user_sk, http_method, \
                            path, headers[http_headers.X_DATE], str(headers[http_headers.EXPIRES]))
    md5.update(all)
    sign = md5.hexdigest()
    return "auth-v1/%s/%s/%s//%s" % (config.user_ak, headers[http_headers.X_DATE],
                                     str(headers[http_headers.EXPIRES]), sign)


def sign(user_ak, user_sk, http_method, path, headers, params, headers_to_sign=None):
    """
    generate sign string
    """
    headers = headers or {}
    params = params or {}

    # 1.generate sign key
    # 1.1. generate auth-string, format string: paddle-auth-v1/{accessKeyId}/{timestamp}/{expirationPeriodInSeconds}
    now_time = int(time.time())
    sign_key_info = 'paddle-auth-v1/%s/%s/%d' % (
        user_ak,
        now_time,
        headers[http_headers.EXPIRES])
    # 1.2. with auth-string as SK, then generate sign key with SHA-256 algorithm
    sign_key = hmac.new(
        user_sk,
        sign_key_info,
        hashlib.sha256).hexdigest()
    # 2.get canonical uri
    canonical_uri = get_canonical_uri(path)

    # 3.get query string
    # print params
    canonical_querystring = get_canonical_querystring(params)

    # 4.get canonical header
    canonical_headers = get_canonical_headers(headers, headers_to_sign)

    # 5.join them, generate a big string
    string_to_sign = '\n'.join(
        [http_method, canonical_uri, canonical_querystring, canonical_headers])
    # print "str2sign " + string_to_sign + "\n"
    # generate sign result with string_to_sign and sign key in step 5 and 1.
    sign_result = hmac.new(sign_key, string_to_sign, hashlib.sha256).hexdigest()
    # 7.splice the final signature string
    if headers_to_sign:
        # design header to sign
        result = '%s/%s/%s' % (sign_key_info, ';'.join(headers_to_sign), sign_result)
    else:
        # return the default signature string when don't design header to sign
        result = '%s//%s' % (sign_key_info, sign_result)

    return result


def normalize_string(in_str, encoding_slash=True):
    """
    Encode in_str.
    When encoding_slash is True, don't encode skip_chars, vice versa.

    :type in_str: string
    :param in_str: None

    :type encoding_slash: Bool
    :param encoding_slash: None
    ===============================
    :return:
        **string**
    """
    tmp = []
    for ch in convert_to_standard_string(in_str):
        if ch == '/' and not encoding_slash:
            tmp.append('/')
        else:
            tmp.append(_NORMALIZED_CHAR_LIST[ord(ch)])
    return ''.join(tmp)


def get_canonical_querystring(params, for_signature=False):
    """
    merge params to uri
    :param params:
    :param for_signature:
    :return:
    """
    if params is None:
        return ''
    result = []
    for k, v in params.items():
        if not for_signature or k.lower != http_headers.AUTHORIZATION.lower():
            if v is None:
                v = ''
            result.append('%s=%s' % (normalize_string(k), normalize_string(v)))
    result.sort()
    return '&'.join(result)


def get_canonical_headers(headers, headers_to_sign=None):
    """
    get canonical header
    """
    headers = headers or {}

    # when no design header_to_sign, default use it:
    #   1.host
    #   2.content-md5
    #   3.content-length
    #   4.content-type
    #   5.all header with start x-paddle-
    if headers_to_sign is None or len(headers_to_sign) == 0:
        headers_to_sign = {"host", "content-md5", "content-length", "content-type"}

    f = lambda (key, value): (key.strip().lower(), str(value).strip())

    result = []
    for k, v in map(f, headers.iteritems()):
        if k.startswith(PADDLE_PREFIX) or k in headers_to_sign:
            result.append("%s:%s" % (normalize_string(k), normalize_string(v)))

    result.sort()
    return '\n'.join(result)


def convert_to_standard_string(input_string):
    """
    Encode a string to utf-8.

    :type input_string: string
    :param input_string: None
    =======================
    :return:
        **string**
    """
    if isinstance(input_string, unicode):
        return input_string.encode(paddlecli.DEFAULT_ENCODING)
    else:
        return str(input_string)


def generate_uuid():
    """
    generate uuid
    """
    return uuid.uuid1()


def is_valid_date(str_date):
    """
    input date format should be 'Y-m-d', return True or False
    :param str_date:
    :return:
    """
    try:
        time.strptime(str_date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def is_valid_time(str_time):
    """
    input time format should be 'H:M:S', return True or False
    :param str_time:
    :return:
    """
    try:
        time.strptime(str_time, "%H:%M:%S")
        return True
    except ValueError:
        return False


def is_valid_wall_time(str_time):
    """
    input time format as 'H+:M:S', return True or False
    :param str_time:
    :return:
    """
    pattern="^[0-9]+:[0-5]?[0-9]:[0-5]?[0-9]$"
    if re.match(pattern, str_time):
        return True
    else:
        return False


def is_valid_str(s):
    """
    verify a str
    """
    if re.match('^[0-9a-zA-z_\-\.]+$', s):
        return True
    else:
        return False
