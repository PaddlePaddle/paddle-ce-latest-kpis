"""
This module defines a retry policy class for paddlecloud tool.
"""

import httplib
from urllib2 import HTTPError


class NoRetryPolicy(object):
    """A policy that never retries."""

    def should_retry(self, error, retries_attempted):
        """Always returns False.

        :param error: ignored
        :param retries_attempted: ignored
        :return: always False
        :rtype: bool
        """
        return False

    def get_delay_before_next_retry_in_millis(self, error, retries_attempted):
        """Always returns 0.

        :param error: ignored
        :param retries_attempted: ignored
        :return: always 0
        :rtype: int
        """
        return 0


class BackOffRetryPolicy(object):
    """A policy that retries with exponential back-off strategy.

    This policy will keep retrying until the maximum number of retries is reached. The delay time
    will be a fixed interval for the first time then 2 * interval for the second, 4 * internal for
    the third, and so on. In general, the delay time will be 2^number_of_retries_attempted*interval.

    When a maximum of delay time is specified, the delay time will never exceed this limit.
    """

    def __init__(self,
                 max_error_retry=3,
                 max_delay_in_millis=20 * 1000,
                 base_interval_in_millis=300):
        """
        :param max_error_retry: the maximum number of retries.
        :type max_error_retry: int
        :param max_delay_in_millis: the maximum of delay time in milliseconds.
        :type max_delay_in_millis: int
        :param base_interval_in_millis: the base delay interval in milliseconds.
        :type base_interval_in_millis: int
        :raise ValueError if max_error_retry or max_delay_in_millis is negative.
        """
        if max_error_retry < 0:
            raise ValueError('max_error_retry should be a non-negative integer.')
        if max_delay_in_millis < 0:
            raise ValueError('max_delay_in_millis should be a non-negative integer.')

        self.max_error_retry = max_error_retry
        self.max_delay_in_millis = max_delay_in_millis
        self.base_interval_in_millis = base_interval_in_millis

    def should_retry(self, error, retries_attempted):
        """Return true if the http client should retry the request.

        :param error: the caught error.
        :type error: Exception
        :param retries_attempted: the number of retries which has been attempted before.
        :type retries_attempted: int
        :return: true if the http client should retry the request.
        :rtype: bool
        """

        # stop retrying when the maximum number of retries is reached
        if retries_attempted >= self.max_error_retry:
            return False

        # always retry on IOError
        if isinstance(error, IOError):
            return True

        # Only retry on a subset of service exceptions
        if isinstance(error, HTTPError):
            if error.code == httplib.INTERNAL_SERVER_ERROR:
                return True
            if error.code == httplib.SERVICE_UNAVAILABLE:
                return True
            if error.code == httplib.GATEWAY_TIMEOUT:
                return True

        return False

    def get_delay_before_next_retry_in_millis(self, error, retries_attempted):
        """Returns the delay time in milliseconds before the next retry.

        :param error: the caught error.
        :type error: Exception
        :param retries_attempted: the number of retries which has been attempted before.
        :type retries_attempted: int
        :return: the delay time in milliseconds before the next retry.
        :rtype: int
        """
        if retries_attempted < 0:
            return 0
        delay_in_millis = (1 << retries_attempted) * self.base_interval_in_millis
        if delay_in_millis > self.max_delay_in_millis:
            return self.max_delay_in_millis
        return delay_in_millis
