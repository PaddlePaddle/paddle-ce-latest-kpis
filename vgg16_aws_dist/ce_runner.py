import argparse
import logging
import sys, os
import numpy as np
import threading
import copy
from aws_runner.client.train_command import TrainCommand

# for ce env ONLY

sys.path.append(os.environ['ceroot'])
from kpi import LessWorseKpi

from aws_runner.client.abclient import Abclient

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    '--key_name', type=str, default="", help="required, key pair name")
parser.add_argument(
    '--security_group_id',
    type=str,
    default="",
    help="required, the security group id associated with your VPC")

parser.add_argument(
    '--vpc_id',
    type=str,
    default="",
    help="The VPC in which you wish to run test")
parser.add_argument(
    '--subnet_id',
    type=str,
    default="",
    help="The Subnet_id in which you wish to run test")

parser.add_argument(
    '--pserver_instance_type',
    type=str,
    default="c5.2xlarge",
    help="your pserver instance type, c5.2xlarge by default")
parser.add_argument(
    '--trainer_instance_type',
    type=str,
    default="p2.8xlarge",
    help="your trainer instance type, p2.8xlarge by default")

parser.add_argument(
    '--task_name',
    type=str,
    default="",
    help="the name you want to identify your job")
parser.add_argument(
    '--pserver_image_id',
    type=str,
    default="ami-da2c1cbf",
    help="ami id for system image, default one has nvidia-docker ready, \
    use ami-1ae93962 for us-east-2")

parser.add_argument(
    '--pserver_command',
    type=str,
    default="",
    help="pserver start command, format example: python,vgg.py,batch_size:128,is_local:yes"
)

parser.add_argument(
    '--trainer_image_id',
    type=str,
    default="ami-da2c1cbf",
    help="ami id for system image, default one has nvidia-docker ready, \
    use ami-1ae93962 for us-west-2")

parser.add_argument(
    '--trainer_command',
    type=str,
    default="",
    help="trainer start command, format example: python,vgg.py,batch_size:128,is_local:yes"
)

parser.add_argument(
    '--availability_zone',
    type=str,
    default="us-east-2a",
    help="aws zone id to place ec2 instances")

parser.add_argument(
    '--trainer_count', type=int, default=1, help="Trainer count")

parser.add_argument(
    '--pserver_count', type=int, default=1, help="Pserver count")

parser.add_argument(
    '--action', type=str, default="create", help="create|cleanup|status")

parser.add_argument('--pem_path', type=str, help="private key file")

parser.add_argument(
    '--pserver_port', type=str, default="5436", help="pserver port")

parser.add_argument(
    '--docker_image', type=str, default="busybox", help="training docker image")

parser.add_argument(
    '--master_server_port', type=int, default=5436, help="master server port")

parser.add_argument(
    '--master_server_public_ip', type=str, help="master server public ip")

parser.add_argument(
    '--master_docker_image',
    type=str,
    default="putcn/paddle_aws_master:latest",
    help="master docker image id")

parser.add_argument(
    '--no_clean_up',
    type=str2bool,
    default=False,
    help="whether to clean up after training")

parser.add_argument(
    '--online_mode',
    type=str2bool,
    default=False,
    help="is client activly stays online")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

train_speed_kpi = LessWorseKpi('train_speed', 0.01)
kpis_to_track = {}

def save_to_kpi(name, val):
    val = float(val)
    if name in kpis_to_track:
        kpi_to_track = kpis_to_track[name]
    else:
        kpi_to_track = LessWorseKpi(name, 0.01)
    kpi_to_track.add_record(np.array(val, dtype='float32'))

class DataCollector(object):
    def __init__(self):
        self.store = []
        self.metric_data_identifier = "**metrics_data: "
    def log_processor(self, msg):
        if (msg.startswith(self.metric_data_identifier)):
            str_msg = msg.replace(self.metric_data_identifier, "")
            metrics_raw = str_msg.split(",")
            for metric in metrics_raw:
                metric_data = metric.split("=")
                if metric_data[0].strip() == "train_speed":
                    self.save(metric_data[1])
    def save(self, val):
        self.store.append(float(val))
    def avg(self):
        return np.average(self.store)

solo_data_collector = DataCollector()
def train_without_pserver(args, lock):
    def log_handler(source, id):
        for line in iter(source.readline, ""):
            logging.info("without pserver:")
            logging.info(line)
            solo_data_collector.log_processor(line)

    args.pserver_count = 0
    args.trainer_count = 1
    trainer_command = TrainCommand(args.trainer_command)
    trainer_command.update({"local":"yes"})
    args.trainer_command = trainer_command.unparse()
    logging.info(args)
    abclient = Abclient(args, log_handler, lock)
    abclient.create()

cluster_data_collector = DataCollector()
def train_with_pserver(args, lock):
    def log_handler(source, id):
        for line in iter(source.readline, ""):
            logging.info("with pserver:")
            logging.info(line)
            cluster_data_collector.log_processor(line)

    logging.info(args)
    abclient = Abclient(args, log_handler, lock)
    abclient.create()

if __name__ == "__main__":
    print_arguments()
    if args.action == "create":
        lock = threading.Lock()
        thread_no_pserver = threading.Thread(
            target=train_without_pserver,
            args=(copy.copy(args), lock,)
        )
        thread_with_pserver = threading.Thread(
            target=train_with_pserver,
            args=(copy.copy(args), lock, )
        )
        thread_no_pserver.start()
        thread_with_pserver.start()
        thread_no_pserver.join()
        thread_with_pserver.join()

        speedup_rate = cluster_data_collector.avg()/solo_data_collector.avg()
        logging.info("speed up rate is "+ str(speedup_rate))

        save_to_kpi("speedup_rate", speedup_rate.item())