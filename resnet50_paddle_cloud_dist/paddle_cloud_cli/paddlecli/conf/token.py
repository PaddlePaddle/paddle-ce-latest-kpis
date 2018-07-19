"""
This package provides the local token manager class.
"""
import argparse
import os
import sys
import paddlecli.conf.config as _config
from paddlecli.lib import util

config_folder = os.path.expanduser('~') + os.sep + '.paddlecli'
paddle_token_file = config_folder + os.sep + 'token'
paddle_config_file = config_folder + os.sep + 'config'
USER_SECTION = 'token'


class TokenManager(object):
    """
    token manager class
    """

    def __init__(self, token_file_path):
        self.file = token_file_path
        self.token_name = None
        self.user_ak = None
        self.user_sk = None

    def save(self):
        """
        save content to local file
        """
        self.check_valid()
        config = _config.PaddleConfigParser()
        config.read(self.file)
        if config.has_section(self.token_name) is False:
            config.add_section(self.token_name)
        config.set(self.token_name, 'user_ak', self.user_ak)
        config.set(self.token_name, 'user_sk', self.user_sk)
        with open(self.file, 'w') as config_file:
            config.write(config_file)

    def check_valid(self):
        """
        check token user configuration valid
        """
        if not str.isalnum(self.user_ak) or not str.isalnum(self.user_sk):
            raise Exception("the user's access key or secret key is illegal, "
                            "please check it.")
        if not util.is_valid_str(self.token_name):
            raise Exception("the token's name is illegal, please check it.")

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

    def set_token_name(self, token_name):
        """
        set user_ak
        """
        if token_name is None or len(token_name.strip()) == 0:
            return False
        self.token_name = token_name.strip()
        return True


def token_add_interactive():
    """
    config token user token info
    """
    _config.init_conf_folder(config_folder)
    token_manager = TokenManager(paddle_token_file)
    token_name_prompt = "input your token name: "
    token_name = _config.read_input(token_name_prompt, True, False)
    token_manager.set_token_name(token_name)
    user_ak_prompt = "input your access key: "
    user_ak = _config.read_input(user_ak_prompt, True, False)
    token_manager.set_user_ak(user_ak)
    user_sk_prompt = "input your secret key: "
    user_sk = _config.read_input(user_sk_prompt, True, False)
    token_manager.set_user_sk(user_sk)
    token_manager.save()
    print "A token has been added successfully."
    return


def current(args):
    """
    return the current token
    """
    config = _config.PaddleConfigParser()
    config.read(paddle_config_file)
    current_token_name = _config.get_option(config, USER_SECTION, 'token_name')
    if current_token_name:
        print "* \033[0;32m%s\033[0m" % current_token_name
        sys.exit(0)
    else:
        print "The current token is Null, " \
              "and you can switch to an existing token by 'token --switch'"
        sys.exit(1)


def query_all_tokens():
    """
    query all tokens
    """
    config = _config.PaddleConfigParser()
    config.read(paddle_config_file)
    current_token_name = _config.get_option(config, USER_SECTION, 'token_name')
    if current_token_name:
        print "* \033[0;32m%s\033[0m" % current_token_name
    token_config = _config.PaddleConfigParser()
    token_config.read(paddle_token_file)
    ss = token_config.sections()
    if current_token_name and current_token_name in ss:
        ss.remove(current_token_name)
    for i in ss:
        print "  " + i


def remove_token(args):
    """
    remove token
    """
    if len(args) != 1:
        raise Exception("argument --remove: expected 1 argument(s)")
    name = args.pop()
    removed_flag = False
    config = _config.PaddleConfigParser()
    config.read(paddle_config_file)
    current_token_name = _config.get_option(config, USER_SECTION, 'token_name')
    if name == current_token_name:
        config.remove_section(USER_SECTION)
        with open(paddle_config_file, 'w') as cf:
            config.write(cf)
        removed_flag = True
    token_config = _config.PaddleConfigParser()
    token_config.read(paddle_token_file)
    ss = token_config.sections()
    if name in ss:
        token_config.remove_section(name)
        with open(paddle_token_file, 'w') as gf:
            token_config.write(gf)
        removed_flag = True

    if removed_flag:
        print "The token '%s' has been removed successfully." % name
        sys.exit(0)
    else:
        print "Oops...The token '%s' not exists, please check it." % name
        sys.exit(1)


def switch_token(args):
    """
    switch token to another
    """
    if len(args) != 1:
        raise Exception("argument --switch: expected 1 argument(s)")
    name = args.pop()
    config = _config.PaddleConfigParser()
    config.read(paddle_config_file)
    current_token_name = _config.get_option(config, USER_SECTION, 'token_name')
    if name == current_token_name:
        print "Switched to token '%s'" % name
        return
    token_config = _config.PaddleConfigParser()
    token_config.read(paddle_token_file)
    ss = token_config.sections()

    if name in ss:
        user_ak = _config.get_option(token_config, name, 'user_ak')
        user_sk = _config.get_option(token_config, name, 'user_sk')
        if config.has_section(USER_SECTION) is False:
            config.add_section(USER_SECTION)
        config.set(USER_SECTION, 'user_ak', user_ak)
        config.set(USER_SECTION, 'user_sk', user_sk)
        config.set(USER_SECTION, 'token_name', name)
        with open(paddle_config_file, 'w') as cf:
            config.write(cf)
        print "Switched to token '%s'" % name
    else:
        print "Oops...The token '%s' not exists, please check it." % name
        sys.exit(1)

    return


class AddToken(argparse.Action):
    """
    add a token into file
    """
    def __call__(self, parser, namespace, values, option_string=None):
        token_add_interactive()
        parser.exit()


class QueryAllTokens(argparse.Action):
    """
    query all tokens in local file
    """
    def __call__(self, parser, namespace, values, option_string=None):
        query_all_tokens()
        parser.exit()


class RemoveToken(argparse.Action):
    """
    remove a token
    """
    def __call__(self, parser, namespace, values, option_string=None):
        remove_token(list(values))
        parser.exit()


class SwitchToken(argparse.Action):
    """
    switch to another token
    """
    def __call__(self, parser, namespace, values, option_string=None):
        switch_token(list(values))
        parser.exit()
