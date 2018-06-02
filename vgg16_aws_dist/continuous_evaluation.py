import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import LessWorseKpi, GreaterWorseKpi

kpis_specs = {
    "speedup": [LessWorseKpi, 0.01],
    "train_speed":[LessWorseKpi, 0.01],
    # "converge_speed":[GreaterWorseKpi, 0.01],
    # "gpu_memory":[GreaterWorseKpi, 0.01],
    # "acc_4passes":[GreaterWorseKpi, 0.01],
}

# each row represets a cluster setting with the following columns
# test_name, batch_size, trainer_count, gpus_per_trainer_count, pserver_count
# disable production cluster config for now
# cluster_specs = [
#    ["mnist", 64, 1, 1, 0],
#    ["mnist", 64, 8, 1, 8],
#    ["mnist", 64, 16, 1, 8],
#    ["mnist", 64, 32, 1, 8],
# ]

cluster_specs = [
    ["vgg", 16, 1, 1, 0],
    ["vgg", 16, 4, 4, 4],
    ["vgg", 16, 7, 8, 7],
]

kpis_map = {}

tracking_kpis = []

def generate_cluster_id(cluster_spec):
    return "_".join(map(str, cluster_spec))
def generate_kpi_id(kpi_name, cluster_spec):
    return kpi_name + "_" + generate_cluster_id(cluster_spec)

for kpi_type_name, (Kpi_class, diff_thre) in kpis_specs.items():
    for cluster_spec in cluster_specs:
        kpi_id = generate_kpi_id(kpi_type_name, cluster_spec)
        the_kpi = Kpi_class(kpi_id, diff_thre)
        tracking_kpis.append(the_kpi)
        kpis_map[kpi_id] = the_kpi