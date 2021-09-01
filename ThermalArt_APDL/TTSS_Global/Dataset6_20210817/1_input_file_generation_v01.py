from collections import OrderedDict
from collections import namedtuple
from itertools import product
import sys
import subprocess
import numpy as np
import shutil
import re
import os
import multiprocessing
from multiprocessing import Process, Queue
import time
import matplotlib.pyplot as plt
from os import getpid

start_time = time.time()
# class RunBuilder():
#     @staticmethod
#     def get_runs(params):
#         Run = namedtuple('Run', params.keys())
#         runs = []
#         for v in product(*params.values()):
#             runs.append(Run(*v))
#         return runs

trial = "cluster_v03"
# params = OrderedDict(
#     Power = [0.01, 0.1, 1, 4, 6, 8, 10, 20, 50],
#     HTC = [1e-9, 1e-7, 2e-7, 1e-6, 5e-6, 1e-5],
#     Die_xy = [4000, 10000, 20000, 30000],
#     Die_z = [10, 100, 200],
#     t_Diel = [1.0, 6.0, 10.0],
#     t_Insulator = [0.01, 0.02, 0.05],
#     K_Diel = [0.00138, 0.0138, 0.138]
# )

params = OrderedDict(
    Total_Power = [500, 1000, 10000, 20000],
    # Power = [0.01, 1, 10, 50],
    HTC = [1e-9, 1e-7, 1e-6, 2e-5],
    Die_xy = [4000, 10000, 20000, 30000],
    Die_z = [10, 100, 200],
    t_Diel = [1.0, 6.0, 10.0],
    t_Insulator = [0.01, 0.02, 0.05],
    K_Diel = [0.00138, 0.0138, 0.138]
)

path = os.getcwd()
dest_fpath = path + f"\One_tile_3D_data_{trial}\\"
os.makedirs(os.path.dirname(dest_fpath), exist_ok=True, mode=0o777)
os.chmod(path + f"\One_tile_3D_data_{trial}", 0o777)

seed_file_folder = os.path.abspath(os.path.join(path, os.pardir))
txt_name = "\commands_TTSS_template_11.txt"
seed_file_name = seed_file_folder + txt_name

for root, dirs, files in os.walk(dest_fpath):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o777)
    for f in files:
        os.chmod(os.path.join(root, f), 0o777)

num_cases = 600
# Power = np.random.randint(low = 1, high=5000, size=(num_cases, 1), dtype=int) * 10**-2
Total_Power = np.random.randint(low = 500, high=20000, size=(num_cases, 1), dtype=int)
HTC = np.random.randint(low = 1, high=20000, size=(num_cases, 1), dtype=int) * 10**-9
Die_xy = np.random.randint(low = 4000, high=30000, size=(num_cases, 1), dtype=int)
Die_z = np.random.randint(low = 10, high=200, size=(num_cases, 1), dtype=int)
t_Diel = np.random.uniform(low=1.0, high=10.0, size=(num_cases, 1))
t_Insulator = np.random.uniform(low=1.0, high=5.0, size=(num_cases, 1)) * 10**-2
K_Diel = np.random.uniform(low=1.38, high=138.0, size=(num_cases, 1)) * 10**-3

# C_scale = np.random.randint(low = 1, high=30, size=(num_cases, 1), dtype=int)

# params = np.hstack((Total_Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel, C_scale))
params = np.hstack((Total_Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel))
print("Power max is: ", params[:, 0].max())
print("Power min is: ", params[:, 0].min())
print("HTC max is: ", params[:, 1].max())
print("HTC min is: ", params[:, 1].min())
print("Die_xy max is: ", params[:, 2].max())
print("Die_xy min is: ", params[:, 2].min())
print("Die_z max is: ", params[:, 3].max())
print("Die_z min is: ", params[:, 3].min())
print("t_Diel max is: ", params[:, 4].max())
print("t_Diel min is: ", params[:, 4].min())
print("t_Insulator max is: ", params[:, 5].max())
print("t_Insulator min is: ", params[:, 5].min())
print("K_Diel max is: ", params[:, 6].max())
print("K_Diel min is: ", params[:, 6].min())

# print("C_scale max is: ", params[:, 7].max())
# print("C_scale min is: ", params[:, 7].min())
C_scale_lst = [1, 5, 10, 20, 30]
print("Parameter generation done")
def create_input_files(params):
    for param in params:
        # Power = float(format(param[0], '.2f'))
        PowerTotal = int(param[0])
        HTC = format(param[1], '.0e')
        Die_xy = int(param[2])
        Die_z = int(param[3])
        t_Diel = float(format(param[4], '.1f'))
        t_Insulator = float(format(param[5], '.3f'))
        K_Diel = float(format(param[6], '.5f'))

        for C_scale in C_scale_lst:
            # C_scale = int(param[7])
            target_folder_name = dest_fpath + f"Q_{PowerTotal}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}_C_{C_scale}\\"
            os.makedirs(os.path.dirname(target_folder_name), exist_ok = True, mode = 0o777)
            target_file_name = target_folder_name + txt_name
            shutil.copyfile(seed_file_name, target_file_name)
            f = open(target_file_name, 'r')
            lines = f.readlines()
            # lines[10] = lines[10][:10] + f'{tile_size}\n' ## tile size
            # lines[24] = lines[24][:10] + f'{int(tile_size/10)}\n' ## grid size
            shift = 1
            lines[10+shift] = lines[10+shift][:13] + f'{Die_xy}' + lines[10+shift][18:] ## Die x length
            lines[11+shift] = lines[11+shift][:13] + f'{Die_xy}' + lines[11+shift][18:] ## Die y length
            lines[12+shift] = lines[12+shift][:13] + f'{Die_z}\n' ## Die z length
            lines[13+shift] = lines[13+shift][:17] + f'{t_Insulator}\n' ## Insulator layer thickness
            lines[14+shift] = lines[14+shift][:12] + f'{t_Diel}\n' ## Dielectric layer thickness
            lines[21+shift] = lines[21+shift][:8] + f'{K_Diel}\n' ## Dielectric layer K
            lines[47+shift] = lines[47+shift][:8] + f'{HTC}\n' ## top HTC
            lines[49+shift] = lines[49+shift][:8] + f'{HTC}\n'## bottom HTC
            lines[52+shift] = lines[52+shift][:10] + f'{PowerTotal}\n' ## power

            lines[32+shift] = lines[32+shift][:8] + f'{C_scale}' + lines[32+shift][9:]  ## C_scale
            # lines[294] = lines[294][:14] + '4' + lines[294][15:]
            f.close()
            f = open(target_file_name, 'w')
            f.writelines(lines)
            f.close()

# pool = multiprocessing.Pool(multiprocessing.cpu_count())
# pool.apply_async(create_input_files, [params])
create_input_files(params)


# def solve(folder, input_file):
#     print("I'm process", getpid())
#     args = [r"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe",
#         "-B",
#         "-dis",
#         # "-mpi INTELMPI", 
#         "-np 8",
#         "-dir", dest_fpath + folder,
#         "-j", f"DCurveModel",
#         '-i', input_file,
#         '-o', "output.txt"
#         ]
#     subprocess.run(args, cwd=os.getcwd())

# if __name__ == '__main__':
#     # pool = multiprocessing.Pool(6)
#     for folder in os.listdir(os.path.dirname(dest_fpath)):
#         print("Currently processing: ", folder)
#         temp_working_dir = dest_fpath + folder + '\\'  
#         input_file = os.path.dirname(temp_working_dir) + txt_name
#         os.chdir(temp_working_dir)
#         print("Directory has been changed to: ", temp_working_dir)
#         print("Solving...")
#         solve(folder, input_file)
#         # pool.apply(solve, [folder, input_file])
#         print("finished processing: ", folder)
#     # pool.close()
#     # pool.join()

#     for folder in os.listdir(os.path.dirname(dest_fpath)):
#         temp_working_dir = dest_fpath + folder + '\\' 
#         decay_curve_file = temp_working_dir + "decay_curve.txt"
#         decay_curve_data = np.loadtxt(decay_curve_file, skiprows = 18)

#         font_size = 15
#         plt.rcParams.update({'font.size': font_size})
#         fig = plt.figure(figsize=(7, 7), dpi = 50)
#         ax = fig.add_subplot(111)
#         ax.set_title(f"Decay curve {folder}")
#         plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
#         ax.plot(decay_curve_data[:, 0], decay_curve_data[:, 1])
#         fig_dir = temp_working_dir + f"fig\\"
#         os.makedirs(os.path.dirname(fig_dir), exist_ok=True, mode=0o777)
#         plt.savefig(fig_dir + f"Decay curve plot")
#         # plt.show()

# print("Time consumed is: ", time.time() - start_time)

# Time consumed is:  0.039886474609375
