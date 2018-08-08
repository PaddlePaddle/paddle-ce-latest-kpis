import json
import os
import sys
import subprocess
import time

def _fetch_job_id(result):
    if "Congratulations" not in result:
        return None

    return result.split("=")[1].strip()

def _wait_paddle_job_success(job_id):
    retry_times = 150
    retry_interval = 60 # secs
    while retry_times:
        try:
            cmd = [ 'paddlecloud', 'job',
                    '--server', os.getenv("PADDLE_CLOUD_SERVER"),
                    '--port', os.getenv("PADDLE_CLOUD_PORT"),
                    '--user-ak', os.getenv("PADDLE_CLOUD_AK"),
                    '--user-sk', os.getenv("PADDLE_CLOUD_SK"),
                    'state', ('%s' % job_id), ]
            result = subprocess.Popen(cmd, stdout = subprocess.PIPE).communicate()[0].strip()
        except Exception as e:
            print("Failed to query paddle cloud job info, errmsg: %s" % e.message)
            return False

        data = json.loads(result)
        if isinstance(data, str):
            print("Failed to query paddle cloud job info, errdata: %s" % data)
            continue

        if data['code'] != 0:
            print("Failed to query paddle cloud job info, errdata: %s" % data)
            continue

        print("State paddle cloud job, job state: %s" % data['data']['jobStatus'])

        if data['data']['jobStatus'] == "fail":
            print("Failed to execute paddle cloud job, errdata: %s" % data)
            return False

        if data['data']['jobStatus'] == "success":
            print("Success to execute paddle cloud job, errdata: %s" % data)
            return True

        time.sleep(retry_interval)

        retry_times -= 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python %s $PADDLE_CLOUD_RET_CODE $PADDLE_CLOUD_RESULT" % sys.argv[0])
        exit(1)

    if sys.argv[1] != '0':
        print("$PADDLE_CLOUD_RET_CODE is not %s" % sys.argv[1])
        exit(1)

    job_id = _fetch_job_id(sys.argv[2])
    print("Get job_id: %s" % job_id)
    with open('./paddle_cloud_job_id', 'w+') as f:
        f.write(str(job_id))
        f.flush()

    if not job_id:
        print("Failed to get job info from $PADDLE_CLOUD_RESULT: %s" % sys.argv[2])
        exit(1)

    if not _wait_paddle_job_success(job_id):
        exit(1)
    else:
        exit(0)

