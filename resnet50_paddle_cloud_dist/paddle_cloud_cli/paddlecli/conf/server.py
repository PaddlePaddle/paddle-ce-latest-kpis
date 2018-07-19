"""
This package provides the server config class.
"""
import argparse
import os
import paddlecli.lib.util as util
import paddlecli.conf.config as _config


config_folder = os.path.expanduser('~') + os.sep + '.paddlecli'
paddle_config_file = config_folder + os.sep + 'config'
SERVER_SECTION = 'server'


class ServerConfig(object):
    """
    server config class, user can config it or not.
    1. config it using --server-conf
    2. when config is null, paddle client will config server info using conf/url_config.py as default
    """

    def __init__(self, config_file_path):
        self.config_path = config_file_path
        config = _config.PaddleConfigParser()
        config.read(config_file_path)
        self.hostname = _config.get_option(config, SERVER_SECTION, 'host')
        self.port = _config.get_option(config, SERVER_SECTION, 'port')

    def save(self):
        """
        save config to local file
        """
        self.check_valid()
        config = _config.PaddleConfigParser()
        config.read(self.config_path)
        if config.has_section(SERVER_SECTION) is False:
            config.add_section(SERVER_SECTION)
        config.set(SERVER_SECTION, 'host', self.hostname)
        config.set(SERVER_SECTION, 'port', self.port)
        with open(self.config_path, 'w') as config_file:
            config.write(config_file)

    def check_valid(self):
        """
        check configuration valid
        """
        if not util.is_valid_str(self.hostname):
            raise Exception("the hostname is illegal, please check it.")
        if not str.isdigit(self.port):
            raise Exception("the port is illegal, please check it.")

    def set_server_host(self, hostname):
        """
        set server hostname
        """
        if hostname is None or len(hostname.strip()) == 0:
            return False
        self.hostname = hostname.strip()
        return True

    def set_server_port(self, port):
        """
        set server port
        """
        if port is None or len(port.strip()) == 0:
            return False
        self.port = port.strip()
        return True


def load_server_config():
    """
    load server config info from paddle config file
    """
    server_config = ServerConfig(paddle_config_file)
    return server_config


def server_config_interactive():
    """
    server config action
    :return:
    """
    _config.init_conf_folder(config_folder)
    server_config = ServerConfig(paddle_config_file)

    # config server hostname
    server_host_prompt = "input remote server host: "
    if server_config.hostname:
        server_host_prompt = "input new server host (Enter, use the old value): "
    host = _config.read_input(server_host_prompt, True, False, server_config.hostname)
    server_config.set_server_host(host)
    # config server port
    server_port_prompt = "input remote server port: "
    if server_config.port:
        server_port_prompt = "input new server port (Enter, use the old value): "
    port = _config.read_input(server_port_prompt, True, True, server_config.port)
    server_config.set_server_port(port)
    server_config.save()
    print "Configure done."
    return


class ServerConfigAction(argparse.Action):
    """
    config server info action
    """
    def __call__(self, parser, namespace, values, option_string=None):
        server_config_interactive()
        parser.exit()
