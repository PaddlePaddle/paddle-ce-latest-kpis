"""
This package provides an cluster config class.
"""
import copy
import json
import os
import paddlecli.conf.config as _config

CLUSTER_SECTION = 'cluster'
CLUSTER_TYPE = 'clusterType'

# default mpi config
DEFAULT_MPI_PRIORITY = 'normal'
DEFAULT_MPI_WALL_TIME = '10:00:00'
DEFAULT_MPI_NODES = '1'
MPI_TYPE = 'mpi'
K8S_TYPE = 'k8s'

MPI_OPTIONS = {
    'mpiPriority': 'normal',
    'mpiWalltime': '10:00:00',
    'mpiNodes': '1',
    'deviceType': 'CPU'
}

K8S_OPTIONS = {
    'deviceType': 'CPU',
    'k8sTrainersCount': '1',
    'k8sCpuCores': '1',
    'k8sMemory': '1Gi',
    'k8sPserversCount': '1',
    'k8sPscpuCores': '1',
    'k8sPsmemory': '1Gi',
    'k8sIsLocal': '0'
}

K8S_PARAMETERS = [
    'mpiNodes',
    'k8sTrainersCount',
    'k8sCpuCores',
    'k8sPserversCount',
    'k8sPscpuCores'
]
INT_PARAMETERS = set(K8S_PARAMETERS)


class ClusterConfig(object):
    """
    cluster config class
    """

    def __init__(self, config_file_path, load=False):
        self.config_path = config_file_path
        self.cluster_type = None
        self.para_dict = {}
        if load:
            self.load(config_file_path)

    def load(self, config_file_path):
        """
        load config from file
        """
        if os.path.exists(config_file_path):
            config = _config.PaddleConfigParser()
            config.read(config_file_path)
            cluster_type = _config.get_option(config, CLUSTER_SECTION, CLUSTER_TYPE)
            for item in config.items(CLUSTER_SECTION):
                self.para_dict[item[0]] = item[1]
            self.set_cluster_type(cluster_type)
            self.check_valid()
        else:
            raise Exception("config file not exists, "
                            "please create it firstly using <config> [options] command")

    def save(self):
        """
        save config to local file
        """
        self.check_valid()
        config = _config.PaddleConfigParser()
        config.add_section(CLUSTER_SECTION)
        for key in self.para_dict:
            config.set(CLUSTER_SECTION, key, self.para_dict[key])
        with open(self.config_path, 'w') as config_file:
            config.write(config_file)

    def check_valid(self):
        """
        check configuration valid
        """
        if self.cluster_type is None:
            raise Exception("please set clusterType!")
        if self.cluster_type == MPI_TYPE:
            if 'mpiServer' not in self.para_dict or 'mpiQueue' not in self.para_dict:
                raise Exception("mpiServer and mpiQueue must be set if choose MPI")

    def set_cluster_type(self, cluster_type):
        """
        set cluster type
        """
        if cluster_type is None or len(cluster_type.strip()) == 0:
            raise Exception("please set clusterType!")
        cluster_type = cluster_type.strip()
        if cluster_type != MPI_TYPE and cluster_type != K8S_TYPE:
            raise Exception("clusterType must be mpi or k8s!")
        self.cluster_type = cluster_type
        self.para_dict['clusterType'] = cluster_type

    def set_parameter(self, key, value):
        """
        set parameter
        """
        if value is None or len(value.strip()) == 0:
            return False
        self.para_dict[key] = value.strip()
        return True

    def update_parameter(self, para):
        """
        update parameter from dict
        """
        if 'clusterType' in para:
            self.set_cluster_type(para['clusterType'])
        self.para_dict.update(para)

    def get_json(self):
        """
        output config with json format
        """
        para = copy.deepcopy(self.para_dict)
        if self.cluster_type == MPI_TYPE:
            for key in MPI_OPTIONS:
                if key not in para:
                    para[key] = MPI_OPTIONS[key]
        else:
            for key in K8S_OPTIONS:
                if key not in para:
                    para[key] = K8S_OPTIONS[key]
        for key in para:
            if key in INT_PARAMETERS:
                para[key] = int(para[key])

        para_str = json.dumps(para)
        return json.loads(para_str)

    def load_from_json(self, json_data):
        """
        load config from json
        """
        for key in json_data:
            self.para_dict[key] = json_data[key]

    def load_from_args(self, args):
        """
        load config from args
        """
        if args.mpi_priority:
            self.para_dict['mpiPriority'] = args.mpi_priority
        if args.mpi_wall_time:
            self.para_dict['mpiWalltime'] = args.mpi_wall_time
        if args.mpi_nodes:
            self.para_dict['mpiNodes'] = args.mpi_nodes
        if args.device_type:
            self.para_dict['deviceType'] = args.device_type
        if args.k8s_trainers_count:
            self.para_dict['k8sTrainersCount'] = args.k8s_trainers_count
        if args.k8s_cpu_cores:
            self.para_dict['k8sCpuCores'] = args.k8s_cpu_cores
        if args.k8s_memory:
            self.para_dict['k8sMemory'] = args.k8s_memory
        if args.k8s_ps_count:
            self.para_dict['k8sPserversCount'] = args.k8s_ps_count
        if args.k8s_ps_cores:
            self.para_dict['k8sPscpuCores'] = args.k8s_cores
        if args.k8s_ps_memory:
            self.para_dict['k8sPsmemory'] = args.k8s_ps_memory
        if args.k8s_not_local:
            self.para_dict['k8sIsLocal'] = args.k8s_not_local
        if args.k8s_gpu_type:
            self.para_dict['k8sGpuType'] = args.k8s_gpu_type
        if args.cluster_name:
            self.para_dict['clusterName'] = args.cluster_name
