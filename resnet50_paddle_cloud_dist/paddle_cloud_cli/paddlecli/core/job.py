"""
This package provides the paddlecloud job commands
"""
import os
import json
import sys
import paddlecli.conf.api as api
import paddlecli.lib.util as util
import paddlecli.conf.config as _config
import paddlecli.core.http.http_client as http_client
import paddlecli.core.http.http_method as http_method
import paddlecli.core.cluster as cluster
from paddlecli.lib.format import format_json
from paddlecli.lib.format import pretty_print
from paddlecli.lib.format import table_print


def build_cluster_config(args):
    """
    build cluster config from args, output json format
    """
    para_dict = {}

    if args.mpi_priority:
        para_dict['mpiPriority'] = args.mpi_priority
    if args.mpi_wall_time:
        if not util.is_valid_wall_time(args.mpi_wall_time):
            raise Exception("invalid value for '--mpi-wall-time'")
        para_dict['mpiWalltime'] = args.mpi_wall_time

    if args.mpi_nodes:
        if args.mpi_nodes <= 0:
            raise Exception("invalid value for '--mpi-nodes'")
        para_dict['mpiNodes'] = args.mpi_nodes

    if args.k8s_trainers_count:
        if args.k8s_trainers_count <= 0:
            raise Exception("invalid value for '--k8s-trainers'")
        para_dict['k8sTrainersCount'] = args.k8s_trainers_count

    if args.k8s_cpu_cores:
        if args.k8s_cpu_cores <= 0:
            raise Exception("invalid value for '--k8s-cpu-cores'")
        para_dict['k8sCpuCores'] = args.k8s_cpu_cores

    if args.k8s_memory:
        para_dict['k8sMemory'] = args.k8s_memory
        memory = args.k8s_memory
        if len(memory) < 3:
            raise Exception("invalid value for '--k8s-memory'")
        unit = memory[-2:]
        if unit not in ['Gi', 'Mi', 'Ki']:
            raise Exception("invalid value for '--k8s-memory'")
        num = memory[:-2]
        if not num.isdigit() or num <= 0:
            raise Exception("invalid value for '--k8s-memory'")

    if args.k8s_ps_num:
        if args.k8s_ps_num <= 0:
            raise Exception("invalid value for '--k8s-ps-num'")
        para_dict['k8sPserversCount'] = args.k8s_ps_num

    if args.k8s_ps_cpu_cores:
        if args.k8s_ps_cpu_cores <= 0:
            raise Exception("invalid value for '--k8s-ps-cores'")
        para_dict['k8sPscpuCores'] = args.k8s_ps_cpu_cores

    if args.k8s_ps_memory:
        memory = args.k8s_ps_memory
        if len(memory) < 3:
            raise Exception("invalid value for '--k8s-ps-memory'")
        unit = memory[-2:]
        if unit not in ['Gi', 'Mi', 'Ki']:
            raise Exception("invalid value for '--k8s-ps-memory'")
        num = memory[:-2]
        if not num.isdigit() or num <= 0:
            raise Exception("invalid value for '--k8s-ps-memory'")
        para_dict['k8sPsmemory'] = args.k8s_ps_memory
    if args.k8s_gpu_cards:
        if args.k8s_gpu_cards <= 0:
            raise Exception("invalid value for '--k8s-gpu-cards'")
        para_dict['k8sGpuCards'] = args.k8s_gpu_cards

    if args.k8s_priority:
        para_dict['k8sPriority'] = args.k8s_priority

    if args.k8s_wall_time:
        if not util.is_valid_wall_time(args.k8s_wall_time):
            raise Exception("invalid value for '--k8s-wall-time'")
        para_dict['k8sWalltime'] = args.k8s_wall_time

    if args.k8s_not_local:
        para_dict['k8sIsLocal'] = 0
        if args.mpi_nodes is None and args.k8s_ps_num is None:
            raise Exception("you should specify the number of mpi nodes or parameter servers")

    if args.k8s_gpu_type:
        para_dict['k8sGpuType'] = args.k8s_gpu_type

    if args.cluster_name:
        para_dict['clusterName'] = args.cluster_name

    string = json.dumps(para_dict)
    return json.loads(string)


def train(args):
    """
    add paddle job to server
    """
    cluster_config = build_cluster_config(args)
    body = dict()
    body['clusterConf'] = cluster_config
    body['jobName'] = args.JOB_NAME
    body['jobType'] = 'paddle'
    body['startCMD'] = args.START_CMD
    body['clusterName'] = args.cluster_name
    cluster_list = cluster.get_cluster_list(args)
    if len(cluster_list) < 1:
        print "Oops...There are no available clusters in the current token."
        sys.exit(1)
    cluster_names = cluster.get_cluster_names(cluster_list)
    if args.cluster_name not in cluster_names:
        print "Oops...The named cluster '%s' not exists." % args.cluster_name
        print "You can choose an appropriate one from the following items:\n"
        tuple_list = cluster.make_tuple_list_with_keys(cluster_list)
        table_print(tuple_list)
        sys.exit(1)
    body['jobAuth'] = args.job_permission
    if args.job_version:
        body['jobVersion'] = args.job_version
        if args.job_version == 'paddle-fluid-custom':
            if not args.version_image_addr:
                raise Exception("please set docker image address "
                        "if choose paddle-fluid-custom version")
            body['versionImageAddr'] = args.version_image_addr

    if not os.path.isfile(args.JOB_CONF):
        raise Exception("the config file " + args.JOB_CONF + " not exists, please check it.")

    with open(args.JOB_CONF, 'r') as jobConf:
        body['jobConf'] = jobConf.read()

    # support multi files to upload (like: before_hook.sh, end_hook.sh, train.py)
    __files_check_and_upload(args.UPLOAD_FILES, args)

    result = None
    try:
        response = http_client.send_request(
            http_method.POST,
            api.JOB_PATH,
            body, None, None, None, args)
        result = response.read()
        json_obj = json.loads(result)
        if args.json:
            print format_json(json_obj)
            return
        if json_obj['code'] == 0:
            print("Congratulations! The new job is ready "
                  "for training and the returned jobId = " + json_obj['data']['jobId'])
            sys.exit(0)
        else:
            print("Oops...The job submission aborted due to error, " + json_obj['message'])
            sys.exit(1)
    except Exception as e:
        sys.stderr.write("%s\n" % result)
        sys.stderr.write("%s\n" % e)
        sys.exit(1)


def __files_check_and_upload(files, config=None):
    """
    job train --files check and upload to paddle cloud server
    :param files:
    :return:
    """
    files_to_upload = set()
    count = 0
    for item in files:
        if os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                if len(files) > _config.PERMIT_UPLOAD_FILES_NUM_LIMITED:
                    raise Exception("Exceeded the maximum number (15) of files, "
                                    "please adjust path files number to upload")
                for i in files:
                    files_to_upload.add(root + os.sep + i)
                    count += 1
        else:
            files_to_upload.add(item)
            count += 1

    if count > _config.PERMIT_UPLOAD_FILES_NUM_LIMITED:
        raise Exception("Exceeded the maximum number (15) of files, "
                        "please adjust path files number to upload")

    for f in files_to_upload:
        if os.path.isfile(f):
            if os.path.getsize(f) > _config.PERMIT_UPLOAD_FILE_SIZE_LIMITED:
                raise Exception("the file '" + f +
                                "' exceeded the maximum size (1MB) to upload")
            filename = os.path.basename(f)
            suffix = filename[filename.rindex(".")+1:]
            if suffix == 'sh' and filename \
                    not in _config.PERMIT_UPLOAD_SHELL_FILE_LIST:
                raise Exception("the shell file '" + filename +
                                "' should be named in [before_hook.sh, end_hook.sh]")
            body = dict()
            body['version'] = 1
            body['fileName'] = filename
            body['type'] = 'new'
            body['content'] = None

            with open(f, 'r') as input_file:
                body['content'] = input_file.read()
            result = None
            try:
                response = http_client.send_request(
                    http_method.POST,
                    api.SAVE_FILE_URL,
                    body, None, None, None, config)
                result = response.read()
                json_obj = json.loads(result)
                if json_obj['code'] == 0:
                    if not config.json:
                        print("uploading " + f)
                else:
                    print("Oops...The file '" +
                          f + "' upload failed due to error, " + json_obj['message'])
                    sys.exit(1)
            except Exception as e:
                sys.stderr.write("%s\n" % result)
                sys.stderr.write("%s\n" % e)
                sys.exit(1)
        else:
            raise Exception("the file '" + f + "' not exists, please check it")


def kill(args):
    """
    kill running job
    """
    body = dict()
    body['jobIds'] = args.JOB_IDS.split(',')
    body['opType'] = 'Kill'
    response = http_client.send_request(
        http_method.PUT,
        api.JOB_PATH,
        body, None, None, None, args)
    obj = json.loads(response.read())
    print format_json(obj)


def items(args):
    """
    list jobs
    """
    params = dict()
    params['type'] = args.type if args.type else 'normal'
    params['page'] = args.page if args.page else 1
    params['maxKeys'] = args.size if args.size else 10
    response = http_client.send_request(
        http_method.GET,
        api.JOB_PATH,
        None, None, params, None, args)
    body = response.read()
    obj = json.loads(body)
    if obj['code'] != 0:
        raise Exception(obj['message'])
    jobs = obj['data']['jobs']
    for job in jobs:
        if "demoFlag" in job:
            del job['demoFlag']

    if args.json:
        print format_json(obj)
    else:
        pretty_print(jobs)


def info(args):
    """
    get one job detail info
    """
    params = dict()
    params['jobId'] = args.JOB_ID
    response = http_client.send_request(
        http_method.GET,
        api.JOB_PATH,
        None, None, params, None, args)
    obj = json.loads(response.read())
    if obj['code'] != 0:
        raise Exception(obj['message'])

    if "jobConf" in obj['data']:
        del obj['data']['jobConf']

    print format_json(obj)


def status(args):
    """
    get one job status
    """
    params = dict()
    params['jobId'] = args.JOB_ID
    response = http_client.send_request(
        http_method.GET,
        api.JOB_STATUS_PATH,
        None, None, params, None, args)
    obj = json.loads(response.read())
    if obj['code'] != 0:
        raise Exception(obj['message'])
    print format_json(obj)


def delete(args):
    """
    delete one job
    """
    body = dict()
    body['jobIds'] = args.JOB_IDS.split(',')
    response = http_client.send_request(
        http_method.DELETE,
        api.JOB_PATH,
        body, None, None, None, args)
    obj = json.loads(response.read())
    print format_json(obj)


def rerun(args):
    """
    rerun one job
    """
    body = dict()
    body['jobIds'] = args.JOB_IDS.split(',')
    body['opType'] = 'Rerun'

    response = http_client.send_request(
        http_method.PUT,
        api.JOB_PATH,
        body, None, None, None, args)
    obj = json.loads(response.read())
    print format_json(obj)


