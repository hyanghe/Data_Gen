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
    print("Available num of cores: ", multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    start_time = time.time()
    txt_name = "/commands_TTSS_template_10.txt"
    # dest_fpath = "/home/hhe/APDL_Workspace/DecayCurveData/One_tile_3D_data_cluster_division400_random/"
    dest_fpath = "./test_w_pool/"
    for folder in os.listdir(os.path.dirname(dest_fpath)):
        print("Currently processing: ", folder)
        temp_working_dir = dest_fpath + folder + '/'  
        input_file = os.path.dirname(temp_working_dir) + txt_name
        # os.chdir(temp_working_dir)
        print("Directory has been changed to: ", temp_working_dir)
        print("Solving...")
        # solve(folder, input_file)
        # pool.apply_async(solve, [folder, input_file])
        args = [r"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe",
                "-B",
                "-dis",
                # "-mpi INTELMPI",
                "-np 48",
                "-dir", dest_fpath + folder,
                "-j", "TTSS_template",
                '-i', input_file,
                '-o', "output.txt"
                ]
        # subprocess.run(args)
        subprocess.Popen(args)

        print("finished processing: ", folder)

# print("Time consumed is: ", time.time() - start_time)

# Time consumed is:  0.039886474609375
