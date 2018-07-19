"""
This module provide the sub-arg parser for job.
"""

import paddlecli
import paddlecli.core.job


def __build_requried_args_for_train_parser(required_parser):
    """
    build required args for job train parser
    """

    required_parser.add_argument(
        '--job-name',
        required=True,
        type=str,
        help="set the job's name",
        dest='JOB_NAME',
        metavar='<ARG>'
    )

    required_parser.add_argument(
        '--job-conf',
        required=True,
        help='specify a config file',
        dest='JOB_CONF',
        metavar='<file>'
    )

    required_parser.add_argument(
        '--job-version',
        choices=['paddle-v2-v0.10',
            'paddle-fluid-v0.12',
            'paddle-fluid-v0.13',
            'paddle-fluid-custom'
            ],
        help='set the version type, default paddle-fluid-v0.13 \
                (paddle-v2-v0.10, paddle-fluid-v0.12, paddle-fluid-v0.13, paddle-fluid-custom)',
        metavar='',
        default='paddle-fluid-v0.13',
        dest='job_version'
    )

    required_parser.add_argument(
        '--cluster-name',
        required=True,
        help='set an existing cluster name',
        metavar='<ARG>',
        dest='cluster_name'
    )

    required_parser.add_argument(
        '--start-cmd',
        required=True,
        type=str,
        help='set the start command (like: "python train.py")',
        dest='START_CMD',
        metavar='<ARG>'
    )

    required_parser.add_argument(
        '--files',
        required=True,
        nargs='+',
        help='specify files to upload, like a shell/python file',
        dest='UPLOAD_FILES',
        metavar='<file>'
    )


def __build_train_parser(parser):
    """
    job train parser
    """
    optional = parser._action_groups.pop()
    parser.set_defaults(func=paddlecli.core.job.train)
    required_parser = parser.add_argument_group('required arguments')
    __build_requried_args_for_train_parser(required_parser)

    optional.add_argument(
        '--permission',
        choices=['private', 'group'],
        help='job access control, default private (private, group)',
        metavar='',
        default='private',
        dest='job_permission'
    )
    optional.add_argument(
        '--json',
        help='output data in json format',
        action='store_true',
        dest='json'
    )
    parser._action_groups.append(optional)

    mpi_parser = parser.add_argument_group('mpi cluster arguments')
    mpi_parser.add_argument(
        '--mpi-priority',
        choices=['low', 'normal', 'high', 'veryhigh'],
        help='set mpi priority, default normal (low, normal, high, veryhigh)',
        metavar='',
        dest='mpi_priority'
    )
    mpi_parser.add_argument(
        '--mpi-wall-time',
        help='set the max running time, (like: HH:mm:ss)',
        metavar='',
        dest='mpi_wall_time'
    )
    mpi_parser.add_argument(
        '--mpi-nodes',
        type=int,
        help='set the number of nodes for mpi, default 1',
        metavar='',
        dest='mpi_nodes'
    )


    __build_k8s_args_for_job_train(parser)


def __build_k8s_args_for_job_train(parser):
    """
    parser add k8s args
    """

    k8s_parser = parser.add_argument_group('k8s cluster arguments')
    k8s_parser.add_argument(
        '--image-addr',
        help='image address should be set if select paddle-fluid-custom version',
        metavar='',
        dest='version_image_addr'
    )
    k8s_parser.add_argument(
        '--k8s-not-local',
        action='store_true',
        help='set to distributed or local mode, default local',
        dest='k8s_not_local'
    )
    k8s_parser.add_argument(
        '--k8s-gpu-type',
        choices=['baidu/gpu_k40', 'baidu/gpu_p40', 'baidu/gpu_v100'],
        help='set the gpu type (baidu/gpu_k40, baidu/gpu_p40, baidu/gpu_v100)',
        metavar='',
        dest='k8s_gpu_type'
    )
    k8s_parser.add_argument(
        '--k8s-gpu-cards',
        type=int,
        help='set the number of gpu cards for each trainer',
        metavar='',
        dest='k8s_gpu_cards'
    )
    k8s_parser.add_argument(
        '--k8s-priority',
        choices=['high', 'normal', 'low'],
        help='set job priority, default normal (high, normal, low)',
        metavar='',
        dest='k8s_priority'
    )
    k8s_parser.add_argument(
        '--k8s-wall-time',
        help='set the max running time (like: HH:mm:ss)',
        metavar='',
        dest='k8s_wall_time'
    )
    k8s_parser.add_argument(
        '--k8s-trainers',
        type=int,
        help='set the number of trainers, default 1',
        metavar='',
        dest='k8s_trainers_count'
    )
    k8s_parser.add_argument(
        '--k8s-cpu-cores',
        type=int,
        help='set the number of cpu cores for each trainer',
        metavar='',
        dest='k8s_cpu_cores'
    )
    k8s_parser.add_argument(
        '--k8s-memory',
        help='set the memory size for each trainer, default 1Gi (unit: Ki, Mi, Gi)',
        metavar='',
        dest='k8s_memory'
    )
    k8s_parser.add_argument(
        '--k8s-ps-num',
        type=int,
        help='set the number of parameter servers',
        metavar='',
        dest='k8s_ps_num'
    )
    k8s_parser.add_argument(
        '--k8s-ps-cores',
        type=int,
        help='set the number of cpu cores for parameter server',
        metavar='',
        dest='k8s_ps_cpu_cores'
    )
    k8s_parser.add_argument(
        '--k8s-ps-memory',
        help='set the memory size for parameter server, default 1Gi (unit: Ki, Mi, Gi)',
        metavar='',
        dest='k8s_ps_memory'
    )


def __build_kill_parser(parser):
    """
    job kill parser
    """
    parser.set_defaults(func=paddlecli.core.job.kill)
    parser.add_argument(
        'JOB_IDS',
        help='set job ids, separates by comma. for example "id1,id2"')


def __build_list_parser(parser):
    """
    job list parser
    """
    parser.set_defaults(func=paddlecli.core.job.items)
    parser.add_argument(
        '--json',
        action='store_true',
        help='output data in the json format',
        dest='json'
    )
    parser.add_argument(
        '--type',
        action='store',
        choices=['normal', 'favorite', 'group'],
        help='list jobs by type, default normal (normal, favorite, group)',
        metavar='<ARG>'
    )
    parser.add_argument(
        '--page',
        action='store',
        type=int,
        help='specify page number, default 1',
        metavar='<ARG>'
    )
    parser.add_argument(
        '--size',
        action='store',
        type=int,
        help='specify the number of items, default 10',
        metavar='<ARG>'
    )


def __build_info_parser(parser):
    """
    job info parser
    """
    parser.set_defaults(func=paddlecli.core.job.info)
    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        'JOB_ID',
        help='set job id: <ID>'
    )


def __build_status_parser(parser):
    """
    job status parser
    """
    parser.set_defaults(func=paddlecli.core.job.status)
    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        'JOB_ID',
        help='set job id: <ID>'
    )


def __build_delete_parser(parser):
    """
    job delete parser
    """
    parser.set_defaults(func=paddlecli.core.job.delete)
    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        'JOB_IDS',
        help='set job ids, separates by comma. for example "id1,id2"'
    )


def __build_rerun_parser(parser):
    """
    job rerun parser
    """
    parser.set_defaults(func=paddlecli.core.job.rerun)
    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        'JOB_IDS',
        help='set job ids, separates by comma. for example "id1,id2"'
    )


def build_job_parser(parser):
    """
    job arg parser builder
    """

    job_parser_dict = dict()
    job_parser_dict["root"] = parser

    parser.add_argument(
        '--server',
        type=str,
        action='store',
        help='specify remote server host',
        metavar='<ARG>',
        dest='server'
    )
    parser.add_argument(
        '--port',
        action='store',
        type=int,
        help='specify remote server port',
        metavar='<ARG>',
        dest='port'
    )
    parser.add_argument(
        '--user-ak',
        action='store',
        type=str,
        help='specify user access key',
        metavar='<ARG>',
        dest='user_ak'
    )
    parser.add_argument(
        '--user-sk',
        action='store',
        type=str,
        help='specify user secret key',
        metavar='<ARG>',
        dest='user_sk'
    )

    sub_parsers = parser.add_subparsers(
        dest='cmd',
        title='Specific job actions',
        metavar='<action> [options]'
    )

    train_parser = sub_parsers.add_parser(
        'train',
        help='add a new job for training',

    )
    job_parser_dict["train"] = train_parser

    status_parser = sub_parsers.add_parser(
        'state',
        help='return the job state'
    )
    job_parser_dict["state"] = status_parser

    list_parser = sub_parsers.add_parser(
        'list',
        help='list available jobs'
    )
    job_parser_dict["list"] = list_parser

    rerun_parser = sub_parsers.add_parser(
        'rerun',
        help='run job once again'
    )
    job_parser_dict["rerun"] = rerun_parser

    info_parser = sub_parsers.add_parser(
        'info',
        help='display the job info'
    )
    job_parser_dict["info"] = info_parser

    kill_parser = sub_parsers.add_parser(
        'kill',
        help='kill the running job'
    )
    job_parser_dict["kill"] = kill_parser

    delete_parser = sub_parsers.add_parser(
        'delete',
        help='delete the remote job'
    )
    job_parser_dict["delete"] = delete_parser

    __build_train_parser(train_parser)
    __build_list_parser(list_parser)
    __build_info_parser(info_parser)
    __build_delete_parser(delete_parser)
    __build_rerun_parser(rerun_parser)
    __build_kill_parser(kill_parser)
    __build_status_parser(status_parser)

    return job_parser_dict
