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

path = os.getcwd()


def solve(folder, input_file):
    print("I'm process", getpid())
    print("input file is: ", input_file)
    print("current workding dir is: ", os.getcwd())
    print("Available num of cores: ", multiprocessing.cpu_count())
    args = [r"/apps/ansys_inc/v202/ansys/bin/mapdl",
        "-B",
        "-dis",
        "-mpi INTELMPI", 
        # "-np {0}".format(multiprocessing.cpu_count()),
        "-np 48",
        "-dir", dest_fpath + folder,
        "-j", "DCurveModel",
        '-i', input_file,
        '-o', "output.txt"
        ]
    print("cwd is: ", dest_fpath + folder)
    # subprocess.call(args, cwd = dest_fpath + folder, shell=True)
    # subprocess.call(args, cwd = dest_fpath + folder, shell=True)
    print("subprocess is: ", subprocess)
    # subprocess.run(args, cwd=os.getcwd())
    p = subprocess.Popen(args, cwd=os.getcwd())
    p.communicate()

if __name__ == '__main__':
    # pool = multiprocessing.Pool(6)
    start_time = time.time()
    txt_name = "/commands_tile_heating_HTC_BC_area_effects_Haiyang_1.txt"
    dest_fpath = "/home/hhe/APDL_Workspace/DecayCurve_20210816/One_tile_3D_data_cluster_division400/"
    for folder in os.listdir(os.path.dirname(dest_fpath)):
        print("Currently processing: ", folder)
        temp_working_dir = dest_fpath + folder + '/'  
        input_file = os.path.dirname(temp_working_dir) + txt_name
        os.chdir(temp_working_dir)
        print("Directory has been changed to: ", temp_working_dir)
        print("Solving...")
        solve(folder, input_file)

        # Delete unuseful files
	files_in_directory = os.listdir(temp_working_dir)
	print('files_in_directory: ', files_in_directory)
	filtered_files = [file for file in files_in_directory if not file.endswith(".txt")]
	print('filtered_files: ', filtered_files)
	for root, dirs, files in os.walk(temp_working_dir):
    		for d in dirs:
        		os.chmod(os.path.join(root, d), 0o777)
    		for f in files:
        		os.chmod(os.path.join(root, f), 0o777)
	for file in filtered_files:
		path_to_file = os.path.join(temp_working_dir, file)
		os.remove(path_to_file)

        print("finished processing: ", folder)

    for folder in os.listdir(os.path.dirname(dest_fpath)):

        temp_working_dir = dest_fpath + folder + '/' 
        print("current dir: ", temp_working_dir)
        decay_curve_file = temp_working_dir + "decay_curve.txt"
        decay_curve_data = np.loadtxt(decay_curve_file, skiprows = 18)

        font_size = 15
        plt.rcParams.update({'font.size': font_size})
        fig = plt.figure(figsize=(7, 7), dpi = 50)
        ax = fig.add_subplot(111)
        ax.set_title("Decay curve {0}".format(folder))
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.85, wspace=0.3, hspace=0.3)
        print("max temp is: ", decay_curve_data[:, 1].max())
        ax.plot(decay_curve_data[:, 0], decay_curve_data[:, 1])
        ax.set_xlim([-5, decay_curve_data[:, 0].max()])
        fig_dir = temp_working_dir + "fig_1/"
        os.makedirs(os.path.dirname(fig_dir), mode=0o777)
        plt.savefig(fig_dir + "Decay curve plot")
        # plt.show()

print("Time consumed is: ", time.time() - start_time)

# Time consumed is:  0.039886474609375
