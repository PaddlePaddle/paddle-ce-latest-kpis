"""
This package provides the paddle cloud config class.
"""
import os
import ConfigParser

HOST = "paddlecloud.baidu-int.com"
PORT = 80
CONNECTION_TIMEOUT_IN_MILLIS = 1000
DEFAULT_SEND_BUF_SIZE = 1024 * 1024
EXPIRES = 1800

PERMIT_UPLOAD_SHELL_FILE_LIST = ['before_hook.sh', 'end_hook.sh']
PERMIT_UPLOAD_FILE_SIZE_LIMITED = 1024 * 1024
PERMIT_UPLOAD_FILES_NUM_LIMITED = 50

# device type
DEFAULT_DEVICE_TYPE = "CPU"
DEVICE_TYPES = (['CPU', 'GPU'])

# default k8s config
DEFAULT_K8S_TRAINERS_COUNT = '1'
DEFAULT_K8S_CPU_CORES = '1'
DEFAULT_K8S_MEMORY = '1Gi'
DEFAULT_K8S_PS_COUNT = '1'
DEFAULT_K8S_PS_CPU_CORES = '1'
DEFAULT_K8S_PS_MEMORY = '1Gi'

config_folder = os.path.expanduser('~') + os.sep + '.paddlecli'
paddle_config_file = config_folder + os.sep + 'config'


def init_conf_folder(path):
    """
    init configuration folder
    """
    if os.path.exists(path):
        if os.path.isdir(path) is False:
            raise Exception("Cannot create directory '%s': File exists. please check ..."
                            % path)
    else:
        os.mkdir(path)


def get_option(config, section_name, option_name):
    """
    get option value from config
    """
    if config.has_option(section_name, option_name):
        return config.get(section_name, option_name)
    else:
        return None


def read_input(instr, is_iter=False, is_int=False, default=None):
    """
    read input information
    param is_iter:if len(input) == 0, input again
    param is_int:if not input.isdigit() input again
    """
    while True:
        input_str = raw_input(instr).strip()
        if len(input_str) > 0:
            if is_int and input_str.isdigit() is False:
                print "[%s] is not valid number, please try it again." % input_str
                continue
            break
        else:
            if default:
                input_str = default
                break
            if is_iter:
                print "input length must > 0, please try it again."
                continue
            else:
                break
    return input_str


class PaddleConfigParser(ConfigParser.SafeConfigParser):
    """
    PaddleConfigParser class, overwrite optionxform
    """
    def __init__(self):
        ConfigParser.SafeConfigParser.__init__(self, defaults=None)

    def optionxform(self, option_str):
        return option_str
