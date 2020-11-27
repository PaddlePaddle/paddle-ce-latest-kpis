# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi

tagspace_epoch_time_cpu_kpi = DurationKpi('tagspace_epoch_time_cpu', 0.02, 0, actived=True)
tagspace_acc_cpu_kpi = AccKpi('tagspace_acc_cpu', 0.08, 0, actived=True)
tagspace_loss_cpu_kpi = DurationKpi('tagspace_loss_cpu', 0.02, 0, actived=True)
tagspace_epoch_time_gpu1_kpi = DurationKpi('tagspace_epoch_time_gpu1', 0.02, 0, actived=True)
tagspace_acc_gpu1_kpi = AccKpi('tagspace_acc_gpu1', 0.08, 0, actived=True)
tagspace_loss_gpu1_kpi = DurationKpi('tagspace_loss_gpu1', 0.02, 0, actived=True)

textcnn_epoch_time_cpu_kpi = DurationKpi('textcnn_epoch_time_cpu', 0.02, 0, actived=True)
textcnn_acc_cpu_kpi = AccKpi('textcnn_acc_cpu', 0.08, 0, actived=True)
textcnn_loss_cpu_kpi = DurationKpi('textcnn_loss_cpu', 0.02, 0, actived=True)
textcnn_epoch_time_gpu1_kpi = DurationKpi('textcnn_epoch_time_gpu1', 0.02, 0, actived=True)
textcnn_acc_gpu1_kpi = AccKpi('textcnn_acc_gpu1', 0.08, 0, actived=True)
textcnn_loss_gpu1_kpi = DurationKpi('textcnn_loss_gpu1', 0.02, 0, actived=True)

textcnn_pretrain_epoch_time_cpu_kpi = DurationKpi('textcnn_pretrain_epoch_time_cpu', 0.02, 0, actived=True)
textcnn_pretrain_acc_cpu_kpi = AccKpi('textcnn_pretrain_acc_cpu', 0.08, 0, actived=True)
textcnn_pretrain_loss_cpu_kpi = DurationKpi('textcnn_pretrain_loss_cpu', 0.02, 0, actived=True)
textcnn_pretrain_epoch_time_gpu1_kpi = DurationKpi('textcnn_pretrain_epoch_time_gpu1', 0.02, 0, actived=True)
textcnn_pretrain_acc_gpu1_kpi = AccKpi('textcnn_pretrain_acc_gpu1', 0.08, 0, actived=True)
textcnn_pretrain_loss_gpu1_kpi = DurationKpi('textcnn_pretrain_loss_gpu1', 0.02, 0, actived=True)

match_pyramid_epoch_time_cpu_kpi = DurationKpi('match_pyramid_epoch_time_cpu', 0.02, 0, actived=True)
match_pyramid_map_cpu_kpi = AccKpi('match_pyramid_map_cpu', 0.08, 0, actived=True)
match_pyramid_epoch_time_gpu1_kpi = DurationKpi('match_pyramid_epoch_time_gpu1', 0.02, 0, actived=True)
match_pyramid_map_gpu1_kpi = AccKpi('match_pyramid_map_gpu1', 0.08, 0, actived=True)

dsmm_pos_neg_cpu_kpi = AccKpi('dsmm_pos_neg_cpu', 0.02, 0, actived=True)
dsmm_pos_neg_gpu1_kpi = AccKpi('dsmm_pos_neg_gpu1', 0.02, 0, actived=True)

multiview_simnet_pos_neg_cpu_kpi = AccKpi('multiview_simnet_pos_neg_cpu', 0.02, 0, actived=True)
multiview_simnet_pos_neg_gpu1_kpi = AccKpi('multiview_simnet_pos_neg_gpu1', 0.02, 0, actived=True)

esmm_epoch_time_cpu_kpi = DurationKpi('esmm_epoch_time_cpu', 0.02, 0, actived=True)
esmm_AUC_ctr_cpu_kpi = AccKpi('esmm_AUC_ctr_cpu', 0.02, 0, actived=True)
esmm_AUC_ctcvr_cpu_kpi = AccKpi('esmm_AUC_ctcvr_cpu', 0.02, 0, actived=True)
esmm_epoch_time_gpu1_kpi = DurationKpi('esmm_epoch_time_gpu1', 0.02, 0, actived=True)
esmm_AUC_ctr_gpu1_kpi = AccKpi('esmm_AUC_ctr_gpu1', 0.02, 0, actived=True)
esmm_AUC_ctcvr_gpu1_kpi = AccKpi('esmm_AUC_ctcvr_gpu1', 0.02, 0, actived=True)

mmoe_epoch_time_cpu_kpi = DurationKpi('mmoe_epoch_time_cpu', 0.02, 0, actived=True)
mmoe_AUC_income_cpu_kpi = AccKpi('mmoe_AUC_income_cpu', 0.02, 0, actived=True)
mmoe_AUC_marital_cpu_kpi = AccKpi('mmoe_AUC_marital_cpu', 0.02, 0, actived=True)
mmoe_epoch_time_gpu1_kpi = DurationKpi('mmoe_epoch_time_gpu1', 0.02, 0, actived=True)
mmoe_AUC_income_gpu1_kpi = AccKpi('mmoe_AUC_income_gpu1', 0.02, 0, actived=True)
mmoe_AUC_marital_gpu1_kpi = AccKpi('mmoe_AUC_marital_gpu1', 0.02, 0, actived=True)

dnn_epoch_time_cpu_kpi = DurationKpi('dnn_epoch_time_cpu', 0.02, 0, actived=True)
dnn_auc_cpu_kpi = AccKpi('dnn_auc_cpu', 0.08, 0, actived=True)
dnn_epoch_time_gpu1_kpi = DurationKpi('dnn_epoch_time_gpu1', 0.02, 0, actived=True)
dnn_auc_gpu1_kpi = AccKpi('dnn_auc_gpu1', 0.08, 0, actived=True)

deepfm_epoch_time_cpu_kpi = DurationKpi('deepfm_epoch_time_cpu', 0.02, 0, actived=True)
deepfm_auc_cpu_kpi = AccKpi('deepfm_auc_cpu', 0.08, 0, actived=True)
deepfm_epoch_time_gpu1_kpi = DurationKpi('deepfm_epoch_time_gpu1', 0.02, 0, actived=True)
deepfm_auc_gpu1_kpi = AccKpi('deepfm_auc_gpu1', 0.08, 0, actived=True)

fm_epoch_time_cpu_kpi = DurationKpi('fm_epoch_time_cpu', 0.02, 0, actived=True)
fm_auc_cpu_kpi = AccKpi('fm_auc_cpu', 0.08, 0, actived=True)
fm_epoch_time_gpu1_kpi = DurationKpi('fm_epoch_time_gpu1', 0.02, 0, actived=True)
fm_auc_gpu1_kpi = AccKpi('fm_auc_gpu1', 0.08, 0, actived=True)

logistic_regression_epoch_time_cpu_kpi = DurationKpi('logistic_regression_epoch_time_cpu', 0.02, 0, actived=True)
logistic_regression_auc_cpu_kpi = AccKpi('logistic_regression_auc_cpu', 0.08, 0, actived=True)
logistic_regression_epoch_time_gpu1_kpi = DurationKpi('logistic_regression_epoch_time_gpu1', 0.02, 0, actived=True)
logistic_regression_auc_gpu1_kpi = AccKpi('logistic_regression_auc_gpu1', 0.08, 0, actived=True)

wide_deep_epoch_time_cpu_kpi = DurationKpi('wide_deep_epoch_time_cpu', 0.02, 0, actived=True)
wide_deep_auc_cpu_kpi = AccKpi('wide_deep_auc_cpu', 0.08, 0, actived=True)
wide_deep_acc_cpu_kpi = AccKpi('wide_deep_acc_cpu', 0.08, 0, actived=True)
wide_deep_epoch_time_gpu1_kpi = DurationKpi('wide_deep_epoch_time_gpu1', 0.02, 0, actived=True)
wide_deep_auc_gpu1_kpi = AccKpi('wide_deep_auc_gpu1', 0.08, 0, actived=True)
wide_deep_acc_gpu1_kpi = AccKpi('wide_deep_acc_gpu1', 0.08, 0, actived=True)

gnn_epoch_time_cpu_kpi = DurationKpi('gnn_epoch_time_cpu', 0.02, 0, actived=True)
gnn_recall20_cpu_kpi = AccKpi('gnn_recall20_cpu', 0.08, 0, actived=True)
gnn_epoch_time_gpu1_kpi = DurationKpi('gnn_epoch_time_gpu1', 0.02, 0, actived=True)
gnn_recall20_gpu1_kpi = AccKpi('gnn_recall20_gpu1', 0.08, 0, actived=True)

word2vec_epoch_time_cpu_kpi = DurationKpi('word2vec_epoch_time_cpu', 0.02, 0, actived=True)
word2vec_acc_cpu_kpi = AccKpi('word2vec_acc_cpu', 0.08, 0, actived=True)
word2vec_epoch_time_gpu1_kpi = DurationKpi('word2vec_epoch_time_gpu1', 0.02, 0, actived=True)
word2vec_acc_gpu1_kpi = AccKpi('word2vec_acc_gpu1', 0.08, 0, actived=True)

tracking_kpis = [
    tagspace_epoch_time_cpu_kpi,
    tagspace_acc_cpu_kpi,
    tagspace_loss_cpu_kpi,
    tagspace_epoch_time_gpu1_kpi,
    tagspace_acc_gpu1_kpi,
    tagspace_loss_gpu1_kpi,
    textcnn_epoch_time_cpu_kpi,
    textcnn_acc_cpu_kpi,
    textcnn_loss_cpu_kpi,
    textcnn_epoch_time_gpu1_kpi,
    textcnn_acc_gpu1_kpi,
    textcnn_loss_gpu1_kpi,
    textcnn_pretrain_epoch_time_cpu_kpi,
    textcnn_pretrain_acc_cpu_kpi,
    textcnn_pretrain_loss_cpu_kpi,
    textcnn_pretrain_epoch_time_gpu1_kpi,
    textcnn_pretrain_acc_gpu1_kpi,
    textcnn_pretrain_loss_gpu1_kpi,
    match_pyramid_epoch_time_cpu_kpi,
    match_pyramid_map_cpu_kpi,
    match_pyramid_epoch_time_gpu1_kpi,
    match_pyramid_map_gpu1_kpi,
    dsmm_pos_neg_cpu_kpi,
    dsmm_pos_neg_gpu1_kpi,
    multiview_simnet_pos_neg_cpu_kpi,
    multiview_simnet_pos_neg_gpu1_kpi,
    esmm_epoch_time_cpu_kpi,
    esmm_AUC_ctr_cpu_kpi,
    esmm_AUC_ctcvr_cpu_kpi,
    esmm_epoch_time_gpu1_kpi,
    esmm_AUC_ctr_gpu1_kpi,
    esmm_AUC_ctcvr_gpu1_kpi,
    mmoe_epoch_time_cpu_kpi,
    mmoe_AUC_income_cpu_kpi,
    mmoe_AUC_marital_cpu_kpi,
    mmoe_epoch_time_gpu1_kpi,
    mmoe_AUC_income_gpu1_kpi,
    mmoe_AUC_marital_gpu1_kpi,
    dnn_epoch_time_cpu_kpi,
    dnn_auc_cpu_kpi,
    dnn_epoch_time_gpu1_kpi,
    dnn_auc_gpu1_kpi,
    deepfm_epoch_time_cpu_kpi,
    deepfm_auc_cpu_kpi,
    deepfm_epoch_time_gpu1_kpi,
    deepfm_auc_gpu1_kpi,
    fm_epoch_time_cpu_kpi,
    fm_auc_cpu_kpi,
    fm_epoch_time_gpu1_kpi,
    fm_auc_gpu1_kpi,
    logistic_regression_epoch_time_cpu_kpi,
    logistic_regression_auc_cpu_kpi,
    logistic_regression_epoch_time_gpu1_kpi,
    logistic_regression_auc_gpu1_kpi,
    wide_deep_epoch_time_cpu_kpi,
    wide_deep_auc_cpu_kpi,
    wide_deep_acc_cpu_kpi,
    wide_deep_epoch_time_gpu1_kpi,
    wide_deep_auc_gpu1_kpi,
    gnn_epoch_time_cpu_kpi,
    wide_deep_acc_gpu1_kpi,
    gnn_recall20_cpu_kpi,
    gnn_epoch_time_gpu1_kpi,
    gnn_recall20_gpu1_kpi,
    word2vec_epoch_time_cpu_kpi,
    word2vec_acc_cpu_kpi,
    word2vec_epoch_time_gpu1_kpi,
    word2vec_acc_gpu1_kpi
]


def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    train_cost\t1.0
    test_cost\t1.0
    train_cost\t1.0
    train_cost\t1.0
    train_acc\t1.2
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            yield kpi_name, kpi_value


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for (kpi_name, kpi_value) in parse_log(log):
        print(kpi_name, kpi_value)
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
