import os
import time
import argparse
import numpy as np

from utils import grid_search as GS


def get_spare_gpu(min_mem):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_info = ''
    for u in gpu_status:
        gpu_info += u
    gpu_info = gpu_info.strip().split('\n')
    available_gpus = []
    for i, info in enumerate(gpu_info):
        info = [u for u in info.split(' ') if len(u) > 0]
        gpu_spare_memory = int(info[8].split('M')[0]) - int(info[6].split('M')[0])
        gpu_util = int(info[9].split('%')[0])
        if gpu_spare_memory > min_mem:
            available_gpus.append((gpu_spare_memory, i))
    available_gpus = [u[1] for u in sorted(available_gpus, reverse=True)]
    return available_gpus


def task_queue(cmd, min_mem, min_gpus=2, interval=5, patience=1000, num_rty=3, use_cpu=False, gpu_assignments=None, rand_flag=False):
    for command in cmd:
        flag = 1  # flag == 0 means the task succeed
        retry = 0
        ori_cmd = command
        while flag != 0:
            device = []
            command = ori_cmd
            if(use_cpu):
                print(' ----- Executing task on CPU ----- ')
            else:
                if not gpu_assignments:
                    device = get_spare_gpu(min_mem)
                    count = 0
                    while len(device) < min_gpus:
                        if count > patience:
                            print(' -------------- Command failed -------------- ')
                            print(command)
                            return 0
                        time.sleep(interval)
                        device = get_spare_gpu(min_mem)
                    if rand_flag:
                        cuda_idx = np.random.choice(device, min_gpus, replace=False)
                    else:
                        cuda_idx = device[:min_gpus]
                    dvs = ''
                    for u in cuda_idx:
                        dvs += str(u) + ','
                    cuda_idx = dvs[:-1]
                else:
                    cuda_idx = gpu_assignments
                cmd_dvs = " --gpus " + cuda_idx
                command = command.strip() + cmd_dvs + '\n'
                print(' ----- Executing task on GPU {} ----- '.format(cuda_idx))
            print(command)

            time.sleep(1)  # Time for manual intervention
            flag = os.system(command)
            flag >>= 8
            if flag:  # flag != 0 means the task failed
                time.sleep(interval)  # If the task fails, wait n seconds, then reapply for resources.
                retry += 1
            if retry >= num_rty:
                print(' -------------- Command failed -------------- ')
                print(command)
                return 0
    return 1


def get_experiments(path, script_name):
    cmd = GS.yaml_to_grid_params(path, script_name)
    return cmd


def get_args():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('-p', '--config_path', type=str,
                        default='configs/test.yaml', help='config path')
    parser.add_argument('-m', '--min_mem', type=int,
                        default=6000, help='min memory needed')
    parser.add_argument('-s', '--script_name', type=str, default='main.py',
                        help='script_name')
    parser.add_argument('-i', '--interval', type=int, default=3,
                        help='interval between initiating two tasks')
    parser.add_argument('-r', '--num_rty', type=int, default=3,
                        help='retry times')
    parser.add_argument('-n', '--num_gpus', type=int, default=2,
                        help='number of gpus used')
    parser.add_argument('--use_cpu', action='store_true',
                        help='use CPU instead of GPU (when testing in PC)')
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('-gpus', '--gpu_assignments', type=str, default='')
    parser.add_argument('--rand_flag', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    cmd = get_experiments(args.config_path, args.script_name)
    task_queue(cmd, args.min_mem, args.num_gpus, args.interval, args.patience, args.num_rty, args.use_cpu, args.gpu_assignments, args.rand_flag)
    print(' -------------- all experiments down! -------------- ')

# import os
# import time
# import argparse
# import numpy as np
#
# from utils import grid_search as GS
#
#
# def get_spare_gpu(min_mem):
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
#     gpu_info = ''
#     for u in gpu_status:
#         gpu_info += u
#     gpu_info = gpu_info.strip().split('\n')
#     prior_gpus = []
#     spare_gpus = []
#     spare_gpus_bak = []
#     for i, gpus in enumerate(gpu_info):
#         gpus = [u for u in gpus.split(' ') if len(u) > 0]
#         # gpu_used_memory = int(gpus[6].split('M')[0])
#         gpu_spare_memory = int(gpus[8].split('M')[0]) - int(gpus[6].split('M')[0])
#         gpu_util = int(gpus[9].split('%')[0])
#         if gpu_spare_memory > min_mem and gpu_util == 0:
#             prior_gpus.append(i)
#         if gpu_spare_memory > min_mem and 0 < gpu_util < 60:
#             spare_gpus.append(i)
#         if gpu_spare_memory > min_mem and gpu_util > 60:
#             spare_gpus_bak.append(i)
#     return prior_gpus, spare_gpus, spare_gpus_bak
#
#
# def task_queue(cmd, min_mem, interval=5, num_rty=3, use_cpu=False):
#     for command in cmd:
#         flag = 1
#         retry = 0
#         ori_cmd = command
#         # flag == 0 means the task succeed
#         while flag != 0:
#             device = []
#             command = ori_cmd
#             if(use_cpu):
#                 print(' ----- Executing task on CPU ----- ')
#             else:
#                 while len(device) == 0:
#                     time.sleep(interval)
#                     device, device_bak1, device_bak2 = get_spare_gpu(min_mem)
#                     if len(device) == 0:
#                         device += device_bak1
#                     if len(device) == 0:
#                         device += device_bak2
#                     if len(device) == 0:
#                         raise NameError('no available cuda device！')
#
#                 cuda_idx = np.random.randint(0, len(device))
#                 cmd_dvs = " --device 'cuda:{}'".format(device[cuda_idx])
#                 command = command.strip() + cmd_dvs + '\n'
#                 print(' ----- Executing task on GPU {} ----- '.format(device[cuda_idx]))
#             time.sleep(1)  # Time for manual intervention
#             flag = os.system(command)
#             flag >>= 8
#             # If the task fails, wait 10 seconds, then reapply for resources. Exit if it fails x times.
#             time.sleep(10)
#             if flag:  # flag != 0 means the task failed
#                 retry += 1
#             if retry >= num_rty:
#                 print(' -------------- Command failed -------------- ')
#                 print(command)
#                 return 0
#     return 0
#
#
# def get_experiments(path, script_name):
#     cmd = GS.yaml_to_grid_params(path, script_name)
#     return cmd
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='Experiments')
#     parser.add_argument('-p', '--config_path', type=str,
#                         default='configs/exp_configs/base_model_param_search_0.yaml', help='config path')
#     parser.add_argument('-m', '--min_mem', type=int,
#                         default=6000, help='min memory needed')
#     parser.add_argument('-s', '--script_name', type=str, default='main.py',
#                         help='script_name')
#     parser.add_argument('-i', '--interval', type=int, default=3,
#                         help='interval between initiating two tasks')
#     parser.add_argument('-r', '--num_rty', type=int, default=3,
#                         help='retry times')
#     parser.add_argument('--use_cpu', action='store_true',
#                         help='use CPU instead of GPU (when testing in PC)')
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#     args = get_args()
#     cmd = get_experiments(args.config_path, args.script_name)
#     task_queue(cmd, args.min_mem, args.interval, args.num_rty, args.use_cpu)
#     print(' -------------- all experiments down! -------------- ')
