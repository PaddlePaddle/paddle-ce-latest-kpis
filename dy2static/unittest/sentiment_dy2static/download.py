#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import time
import shutil
import requests
import sys
import tarfile
import zipfile
import platform
import functools

lasttime = time.time()
FLUSH_INTERVAL = 0.1

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def get_platform():
    return platform.platform()


def is_windows():
    return get_platform().lower().startswith("windows")


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def _download_file(url, savepath, print_progress):
    r = requests.get(url, stream=True)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            starttime = time.time()
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)


def _uncompress_file(filepath, extrapath, delete_file, print_progress):
    if print_progress:
        print("Uncompress %s" % os.path.basename(filepath))

    if filepath.endswith("zip"):
        handler = _uncompress_file_zip
    elif filepath.endswith("tgz"):
        handler = _uncompress_file_tar
    else:
        handler = functools.partial(_uncompress_file_tar, mode="r")

    for total_num, index, rootpath in handler(filepath, extrapath):
        if print_progress:
            done = int(50 * float(index) / total_num)
            progress("[%-50s] %.2f%%" %
                     ('=' * done, float(100 * index) / total_num))
    if print_progress:
        progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

    if delete_file:
        os.remove(filepath)

    return rootpath


def _uncompress_file_zip(filepath, extrapath):
    files = zipfile.ZipFile(filepath, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    total_num = len(filelist)
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def _uncompress_file_tar(filepath, extrapath, mode="r:gz"):
    files = tarfile.open(filepath, mode)
    filelist = files.getnames()
    total_num = len(filelist)
    rootpath = filelist[0]
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def download_file_and_uncompress(url,
                                 savepath=None,
                                 extrapath=None,
                                 extraname=None,
                                 print_progress=True,
                                 cover=False,
                                 delete_file=True):
    if savepath is None:
        savepath = "."

    if extrapath is None:
        extrapath = "."

    savename = url.split("/")[-1]
    savepath = os.path.join(savepath, savename)
    savename = ".".join(savename.split(".")[:-1])
    savename = os.path.join(extrapath, savename)
    extraname = savename if extraname is None else os.path.join(extrapath,
                                                                extraname)

    if cover:
        if os.path.exists(savepath):
            shutil.rmtree(savepath)
        if os.path.exists(savename):
            shutil.rmtree(savename)
        if os.path.exists(extraname):
            shutil.rmtree(extraname)

    if not os.path.exists(extraname):
        if not os.path.exists(savename):
            if not os.path.exists(savepath):
                _download_file(url, savepath, print_progress)
            savename = _uncompress_file(savepath, extrapath, delete_file,
                                        print_progress)
            savename = os.path.join(extrapath, savename)


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


def get_uncompress_data(data_path, flag_value=1):
    url = 'https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz'
    if check_data_done(data_path, flag_value) == False:
        print('Will download and express.')
        os.system('rm -rf %s' % data_path)
        os.system('mkdir -p %s' % data_path)
        data_save_path = data_path
        data_extra_path = data_path
        download_file_and_uncompress(
            url=url, savepath=data_save_path, extrapath=data_extra_path)
        flag_file = '%s/done' % data_path
        with open(flag_file, 'w') as f:
            f.write('%s\n' % str(flag_value))
