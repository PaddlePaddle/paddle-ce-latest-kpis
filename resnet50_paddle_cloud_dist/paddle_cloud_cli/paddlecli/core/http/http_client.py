"""
This module provides http request function for paddlecloud tools.
"""

import httplib
import sys
import time
import traceback
import json

import paddlecli
import paddlecli.conf.config as _config
from paddlecli.conf.user import load_user_config
from paddlecli.conf.server import load_server_config
from paddlecli.core.retry_policy import BackOffRetryPolicy
import paddlecli.core.http.http_headers as http_headers
from paddlecli.lib import util


def _send_http_request(conn, http_method, uri, headers, body):
    """
    send http request, put headers and send body data
    :param conn:
    :param http_method:
    :param uri:
    :param headers:
    :param body:
    :return:
    """
    conn.putrequest(http_method, uri, skip_host=True, skip_accept_encoding=True)

    for k, v in headers.items():
        k = util.convert_to_standard_string(k)
        v = util.convert_to_standard_string(v)
        conn.putheader(k, v)
    conn.endheaders()
    if body:
        conn.send(body)

    return conn.getresponse()


def check_headers(headers):
    """
    check value in headers, if \n in value, raise exception
    """
    for k, v in headers.iteritems():
        if isinstance(v, (str, unicode)) and '\n' in v:
            raise Exception(r'There should not be any "\n" in header[%s]:%s' % (k, v))


def send_request(http_method, path, body, headers, params, retry_policy=None, config=None):
    """
    Send request to paddle cloud services.
    """

    server_config = load_server_config()
    server_host = server_config.hostname
    server_port = server_config.port

    if not server_host or not server_port:
        server_host = _config.HOST
        server_port = _config.PORT

    if config:
        if config.user_ak and config.user_sk:
            user_ak = config.user_ak
            user_sk = config.user_sk
        else:
            user_config = load_user_config()
            user_ak = user_config.user_ak
            user_sk = user_config.user_sk

        if config.server and config.port:
            server_host = config.server
            server_port = config.port
    else:
        user_config = load_user_config()
        user_ak = user_config.user_ak
        user_sk = user_config.user_sk

    headers = headers or {}
    headers['content-type'] = 'application/json'
    user_agent = 'paddle-cli/%s/%s/%s' % (paddlecli.CLI_VERSION, sys.version, sys.platform)
    user_agent = user_agent.replace('\n', '')
    headers[http_headers.USER_AGENT] = user_agent
    if http_headers.REQUEST_ID not in headers:
        headers[http_headers.REQUEST_ID] = util.generate_uuid()
    if http_headers.EXPIRES not in headers:
        headers[http_headers.EXPIRES] = _config.EXPIRES

    headers[http_headers.HOST] = server_host

    if isinstance(body, str):
        headers[http_headers.CONTENT_LENGTH] = len(body)
    elif isinstance(body, dict):
        body = json.dumps(body)
        headers[http_headers.CONTENT_LENGTH] = len(body)
    else:
        headers[http_headers.CONTENT_LENGTH] = 0

    encoded_params = util.get_canonical_querystring(params, False)
    if len(encoded_params) > 0:
        uri = path + '?' + encoded_params
    else:
        uri = path
    check_headers(headers)
    if retry_policy is None:
        retry_policy = BackOffRetryPolicy()

    retries_attempted = 0
    errors = []
    while True:
        conn = None
        try:
            headers[http_headers.AUTHORIZATION] = util.sign(
                user_ak,
                user_sk,
                http_method,
                path,
                headers,
                params
            )
            conn = httplib.HTTPConnection(
                server_host,
                server_port,
                _config.CONNECTION_TIMEOUT_IN_MILLIS
            )
            http_response = _send_http_request(conn, http_method, uri, headers, body)
            return http_response
        except Exception as e:
            if conn is not None:
                conn.close()
            # insert ">>>>" before all trace back lines and then save it
            errors.append('\n'.join('>>>>' + line for line in traceback.format_exc().splitlines()))
            if retry_policy.should_retry(e, retries_attempted):
                delay_in_millis = retry_policy.get_delay_before_next_retry_in_millis(
                    e, retries_attempted)
                time.sleep(delay_in_millis / 1000.0)
            else:
                raise Exception('Unable to execute HTTP request. '
                                'Retried %d times. All trace backs:\n%s'
                                % (retries_attempted, '\n'.join(errors)))

        retries_attempted += 1
