import argparse
import logging
import sys, os
import numpy as np
import threading
import copy
import csv
from aws_runner.client.train_command import TrainCommand

# for ce env ONLY

sys.path.append(os.environ['ceroot'])
from continuous_evaluation import cluster_specs, kpis_map, generate_kpi_id, generate_cluster_id

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

class DataCollector(object):
    _instance_store = {}
    @classmethod
    def get_instance_by_spec(cls, cluster_spec):
        cluster_id = generate_cluster_id(cluster_spec)
        if cluster_id not in cls._instance_store:
            cls._instance_store[cluster_id] = cls(cluster_spec)
        return cls._instance_store[cluster_id]
    @classmethod
    def persist_all(cls):
        for _, collector in cls._instance_store.iteritems():
            collector.persist()
    @classmethod
    def generate_csv(cls):
        with open("report.csv", "w") as csvfile:
            fieldnames = []
            rows = []
            for cluster_id, collector in cls._instance_store.iteritems():
                row = {
                    "cluster_spec": cluster_id
                }
                for metric_name, _ in collector.store.iteritems():
                    if metric_name not in fieldnames:
                        fieldnames.append(metric_name)
                    row[metric_name] = collector.avg(metric_name)
                    rows.append(row)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    def __init__(self, cluster_spec):
        self.store = {}
        self.metric_data_identifier = "**metrics_data: "
        self.cluster_spec = cluster_spec
        self.cluster_id = generate_cluster_id(cluster_spec)
    def log_processor(self, source, log_type):
        for msg in iter(source.readline, ""):
            logging.info(self.cluster_id)
            logging.info(msg)
            if (msg.startswith(self.metric_data_identifier)):
                logging.info("metric data found, parse and save it")
                str_msg = msg.replace(self.metric_data_identifier, "")
                metrics_raw = str_msg.split(",")
                for metric in metrics_raw:
                    metric_data = metric.split("=")
                    self.save(metric_data[0], metric_data[1])
    def save(self, key, val):
        key = key.strip()
        if isinstance(val, str):
            val = val.strip()
        if (key not in self.store):
            self.store[key] = []
        logging.info("going to save " + key + "=" + str(val) + "from " + self.cluster_id)
        self.store[key].append(float(val))
    def get(self, key):
        if (key in self.store):
            return self.store[key]
        return None
    def avg(self, key):
        vals = self.get(key)
        if vals is None:
            return None
        return sum(vals)/float(len(vals))
    def persist(self):
        for metric_name, _ in self.store.iteritems():
            kpi_id = generate_kpi_id(metric_name, self.cluster_spec)
            logging.info("going to persist kpi " + kpi_id)
            if kpi_id in kpis_map:
                kpi_instance = kpis_map[kpi_id]
                kpi_instance.add_record(np.array(self.avg(metric_name), dtype='float32'))
                kpi_instance.persist()
                logging.info("done persisting kpi " + kpi_id)
            else:
                logging.info("no such kpi id found in map!!!")
                logging.info(kpi_id)

def train_with_spec(spec, args, lock):
    logging.info("updating cluster config and starting client")
    test_name = spec[0]
    batch_size = spec[1]
    args.trainer_count = spec[2]
    gpus_per_trainer_count = spec[3]
    args.pserver_count = spec[4]
    trainer_command = TrainCommand(args.trainer_command)

    command_to_update = {
        "model": test_name,
        "batch_size": str(batch_size),
        "gpus": str(gpus_per_trainer_count),
    }

    if args.pserver_count == 0 and args.trainer_count == 1:
        command_to_update["update_method"] = "local"
    ''' not yet supported because aws runner can't provide PADDLE_TRAINER_IPS
    if args.pserver_count == 0 and args.trainer_count > 1:
        command_to_update["update_method"] = "nccl2"
    '''

    trainer_command.update(command_to_update)
    args.trainer_command = trainer_command.unparse()
    args.pserver_command = args.trainer_command

    data_collector = DataCollector.get_instance_by_spec(spec)

    logging.info(args)
    abclient = Abclient(args, data_collector.log_processor, lock)
    abclient.create()

'''
ClusterIterator relies on spec structure as follows
 batch_size, trainer_count, gpus_per_trainer_count, pserver_count
 cluster_specs = [
    [64, 1, 1, 0],
    [64, 8, 1, 8],
    [64, 16, 1, 8],
    [64, 32, 1, 8],
 ]
 it will sequentially distribute specs into chunks and make sure each chunk
 does not exceeds trainer and pserver count limit
 above specs will be distributed into 2 chunks
[[64, 1, 1, 0], [64, 8, 1, 8]]
and 
[[64, 16, 1, 8]]

[64, 32, 1, 8] itself does not fit in a single chunk, thus gets discard

'''
class ClusterIterator:
    def __init__(self, specs, trainer_count_threshold = 32, pserver_count_threshold = 10):
        self.specs = specs
        self.trainer_count_threshold = trainer_count_threshold
        self.pserver_count_threshold = pserver_count_threshold
        self.bad_specs = []
    def __iter__(self):
        return self
    def spec_can_not_fit(self, trainer_count, pserver_count):
        return (trainer_count > self.trainer_count_threshold or pserver_count > self.pserver_count_threshold)
    def next(self):
        specs_to_ret = []
        trainer_count = 0
        pserver_count = 0
        if len(self.specs) == 0:
            raise StopIteration()
        else:
            while len(self.specs) != 0:
                next_spec = self.specs[0]
                # when single spec can't even fit, move it to bad spec list
                if self.spec_can_not_fit(next_spec[2], next_spec[4]):
                    self.bad_specs.append(self.specs.pop(0))
                    continue
                trainer_count += next_spec[2]
                pserver_count += next_spec[4]
                if self.spec_can_not_fit(trainer_count, pserver_count):
                    break
                specs_to_ret.append(self.specs.pop(0))
        if len(specs_to_ret) == 0:
            if len(self.bad_specs) != 0:
                logging.info("%d specs not be able to fit in any test chunk" % len(self.bad_specs))
            raise StopIteration()
        return specs_to_ret

if __name__ == "__main__":
    print_arguments()
    if args.action == "create":
        lock = threading.Lock()
        cluster_specs_origin = copy.copy(cluster_specs)
        for specs in ClusterIterator(cluster_specs):
            logging.info("starting a new chunk of test")
            testing_threads = []
            for cluster_spec in specs:
                logging.info("creating cluster thread with spec")
                logging.info(cluster_spec)
                thread = threading.Thread(
                    target=train_with_spec,
                    args=(cluster_spec, copy.copy(args), lock,)
                )
                testing_threads.append(thread)

            for testing_thread in testing_threads:
                testing_thread.start()
            
            for testing_thread in testing_threads:
                testing_thread.join()
            logging.info("testing chunk ended")
        
        logging.info("all testing ended")
        
        # generate speedup rate
        # 0 spec is the baseline
        def get_speed_and_collector_by_spec(spec):
            data_collector = DataCollector.get_instance_by_spec(spec)
            return data_collector.avg("train_speed"), data_collector

        logging.info("generating speedup")

        # base_speed supposed to be one trainer, one gpu, local mode
        base_speed, _ = get_speed_and_collector_by_spec(cluster_specs_origin[0])
        if base_speed is not None:
            logging.info("base speed is %f" % base_speed)
            if base_speed is not None:
                for cluster_spec in cluster_specs_origin:
                    speed, data_collector = get_speed_and_collector_by_spec(cluster_spec)
                    if speed is not None:
                        # speed * trainer_count * gpus_per_trainer_count / base_speed
                        data_collector.save("speedup", speed*cluster_spec[2]*cluster_spec[3]/base_speed)
        else:
            logging.info("base speed is not available")

        DataCollector.persist_all()
        # DataCollector.generate_csv()

