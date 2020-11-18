# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import sys
import zipfile
import logging

from paddle.dataset.common import download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = {
    'coco': [
        # coco2014
        (
            'http://images.cocodataset.org/zips/train2014.zip',
            '0da8c0bd3d6becc4dcb32757491aca88', ),
        (
            'http://images.cocodataset.org/zips/val2014.zip',
            'a3d79f5ed8d289b7a7554ce06a5782b3', ),
        (
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
            '0a379cfc70b0e71301e0f377548639bd', ),
    ],
}


def download_decompress_file(data_dir, url, md5):
    logger.info("Downloading from {}".format(url))
    zip_file = download(url, data_dir, md5)
    logger.info("Decompressing {}".format(zip_file))
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(path=data_dir)
    os.remove(zip_file)


def check_data_done(data_path, flag_value):
    if (os.path.exists(data_path) == False):
        return False
    flag_file = '%s/done' % data_path
    if (os.path.isfile(flag_file) == False):
        return False
    with open(flag_file, 'r') as f:
        rd = f.readlines()
    if (len(rd) != 1):
        return False
    return rd[0].strip() == str(flag_value)


if __name__ == "__main__":
    data_dir = osp.join(osp.expanduser("~"), ".cache/paddle/dataset/coco")
    flag_value = 233
    if check_data_done(data_dir, flag_value) == False:
        print('Will download and express.')
        for name, infos in DATASETS.items():
            for info in infos:
                download_decompress_file(data_dir, info[0], info[1])
            logger.info("Download dataset {} finished.".format(name))
        flag_file = '%s/done' % data_dir
        with open(flag_file, 'w') as f:
            f.write('%s\n' % str(flag_value))
