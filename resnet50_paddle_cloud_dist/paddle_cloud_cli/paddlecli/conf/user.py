"""
This package provides the user config class.
"""
import argparse
import os
import paddlecli.lib.util as util
import paddlecli.conf.config as _config

config_folder = os.path.expanduser('~') + os.sep + '.paddlecli'
paddle_config_file = config_folder + os.sep + 'config'
paddle_token_file = config_folder + os.sep + 'token'
USER_SECTION = 'token'


class UserConfig(object):
    """
    user config class
    """

    def __init__(self, config_file_path, load=False):
        self.config_path = config_file_path
        self.token_name = None
        self.user_ak = None
        self.user_sk = None
        if os.path.exists(config_file_path):
            config = _config.PaddleConfigParser()
            config.read(config_file_path)
            self.token_name = _config.get_option(config, USER_SECTION, 'token_name')
            self.user_ak = _config.get_option(config, USER_SECTION, 'user_ak')
            self.user_sk = _config.get_option(config, USER_SECTION, 'user_sk')

        if load:
            if not os.path.exists(config_file_path):
                raise Exception("the current user's token not exists, "
                                "please create it using 'paddlecloud config -t'")
            self.check_valid()

    def save(self):
        """
        save config to local file
        """
        self.check_valid()
        # write to paddle config file
        config = _config.PaddleConfigParser()
        config.read(self.config_path)
        if config.has_section(USER_SECTION) is False:
            config.add_section(USER_SECTION)
        config.set(USER_SECTION, 'token_name', self.token_name)
        config.set(USER_SECTION, 'user_ak', self.user_ak)
        config.set(USER_SECTION, 'user_sk', self.user_sk)
        with open(self.config_path, 'w') as cf:
            config.write(cf)
        # write to paddle token file
        token_config = _config.PaddleConfigParser()
        token_config.read(paddle_token_file)
        if token_config.has_section(self.token_name) is False:
            token_config.add_section(self.token_name)
        token_config.set(self.token_name, 'user_ak', self.user_ak)
        token_config.set(self.token_name, 'user_sk', self.user_sk)
        with open(paddle_token_file, 'w') as tf:
            token_config.write(tf)

    def check_valid(self):
        """
        check user configuration valid
        """
        if self.user_ak is None or self.user_sk is None or \
                self.token_name is None:
            raise Exception("the current token is illegal, please check it.")
        if not str.isalnum(self.user_ak) or not str.isalnum(self.user_sk):
            raise Exception("the user's access key or secret key is illegal, "
                            "please check it.")
        if self.token_name and not util.is_valid_str(self.token_name):
            raise Exception("the token's name is illegal, please check it.")

    def set_token_name(self, token_name):
        """
        set user_ak
        """
        if token_name is None or len(token_name.strip()) == 0:
            return False
        self.token_name = token_name.strip()
        return True

    def set_user_ak(self, user_ak):
        """
        set user_ak
        """
        if user_ak is None or len(user_ak.strip()) == 0:
            return False
        self.user_ak = user_ak.strip()
        return True

    def set_user_sk(self, user_sk):
        """
        set user_sk
        """
        if user_sk is None or len(user_sk.strip()) == 0:
            return False
        self.user_sk = user_sk.strip()
        return True


def load_user_config():
    """
    load user config info from paddle config file
    """
    user_config = UserConfig(paddle_config_file, True)
    return user_config


def user_config_interactive():
    """
    config user info function
    """
    _config.init_conf_folder(config_folder)
    user_config = UserConfig(paddle_config_file)
    # config token name
    user_ak_prompt = "input your token name: "
    if user_config.token_name:
        user_ak_prompt = "input new token name (enter then use the old one): "
    token_name = _config.read_input(user_ak_prompt, True, False, user_config.token_name)
    user_config.set_token_name(token_name)
    # config user_ak
    user_ak_prompt = "input your access key: "
    if user_config.user_ak:
        user_ak_prompt = "input new access key (enter then use the old one): "
    user_ak = _config.read_input(user_ak_prompt, True, False, user_config.user_ak)
    user_config.set_user_ak(user_ak)
    # config user_sk
    user_sk_prompt = "input your secret key: "
    if user_config.user_sk:
        user_sk_prompt = "input new secret key (enter then use the old one): "

    user_sk = _config.read_input(user_sk_prompt, True, False, user_config.user_sk)
    user_config.set_user_sk(user_sk)
    user_config.save()
    print "Configure done."
    return


def help(args):
    """
    print the help info
    """
    print "usage: paddlecloud " + args.srv + " [-h] [--token] [--server]"


class UserConfigAction(argparse.Action):
    """
    config user info action
    """
    def __call__(self, parser, namespace, values, option_string=None):
        user_config_interactive()
        parser.exit()
