from zipfile import ZipFile
from SALib.sample import sobol_sequence
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import re
import math
import time

def get_all_file_paths(directory):
    file_paths = []
    file_directories = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
        file_directories.append(directories)
    return file_paths, file_directories


def main():
    # path to folder which needs to be zipped
    # directory = './One_tile_3D_data'
    # directory = './One_tile_3D_data_grid_size_one_batch_run_51200'
    directory = './One_tile_3D_data_cluster_division400_random'

    save_data_path = './Data_20210120/One_tile_3D_data/'

    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    # print("The length of file_paths is: ", len(file_paths))
    file_directories = file_dir[0]
    counter = 1
    params_lst = np.empty((0, 7))
    for folder in file_directories:
        source_dir = directory + '/' + folder + '/'
        # os.chdir(temp_working_dir)
        txt_name = "commands_tile_heating_HTC_BC_area_effects.txt"
        source_file_name = source_dir + txt_name
        f = open(source_file_name, 'r')
        lines = f.readlines()
        # tile_size = float(re.findall("\d+", lines[10].split()[0])[0])
        Die_xy = int(re.findall("\d+", lines[5].split()[0])[0])
        Die_z = int(re.findall("\d+", lines[7].split()[0])[0])

        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        t_Insulator = float(re.findall(match_number, lines[8].split()[0])[0])
        t_Diel = float(re.findall(match_number, lines[9].split()[0])[0])
        K_Diel = float(re.findall(match_number, lines[28].split()[1])[0])


        top_HTC = float(re.findall(match_number, lines[44].split()[0])[0])
        bottom_HTC = float(re.findall(match_number, lines[44].split()[0])[0])
        Power = float(re.findall(match_number, lines[48].split()[0])[0])
        HTC = top_HTC



        # target_folder_name = save_data_path + f"TILE_SIZE_{int(tile_size)}_Q_{int(Power)}_HTC_{HTC}/"
        target_folder_name = save_data_path + f"Q_{Power}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}\\"
        params = np.asarray([Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel])
        params_lst = np.vstack((params_lst, params))

    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    markers = ['.', 'x', '^', '*', 'o', 'v', '<', '>', ',']
    legend = ['power: 1e-2--50', 'HTC (log scale): 1e-9--2e-5', 'Die_xy: 4000--30000',\
              'Die_z: 10--200', 't_Diel: 1--10', 't_Insulator: 1e-2--5e-2', 'K_Diel: 138e-5--138e-3']
    val = 0.  # this is the value where you want the data to appear on the y-axis.
    plt.rcParams.update({'font.size': 22})
    for j in range(params_lst.shape[1]):
        # val += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ar = params_lst[:, j] # just as an example array
        if j == 1:
            ax.plot(np.log10(ar), np.zeros_like(ar) + val, colors[j] + markers[j])
        else:
            ax.plot(ar, np.zeros_like(ar) + val, colors[j] + markers[j])
        plt.legend([legend[j]])
        plt.savefig(save_data_path + f'{j}')
        # plt.show()


    print('done')
        # decay_curve_data_txt_name = "decay_curve.txt"
        # decay_curve_data_file_name = source_dir + decay_curve_data_txt_name
        #
        # decay_f = open(decay_curve_data_file_name, 'r').readlines()
        #
        # start = 1002
        # end = 1014
        # del decay_f[start:end]
        # for i in range(1, len(decay_f)//999):
        #     start = start + 999 - 13
        #     end = end + 999 - 13
        #     del decay_f[start:end+1]
        # del decay_f[:15]
        #
        # if not os.path.exists(target_folder_name):
        #     os.makedirs(target_folder_name, exist_ok=True, mode=0o777)
        #
        # decay_curve_data_file_name_updated = os.path.dirname(target_folder_name) + "/decay_curve_filtered.txt"
        # decay_f_v01 = open(decay_curve_data_file_name_updated, "w+")
        # for line in decay_f:
        #     decay_f_v01.write(line)
        # decay_f_v01.close()
        # decay_curve_data = np.loadtxt(decay_curve_data_file_name_updated)
        # boundary_point = decay_curve_data[-1][0]
        #
        # decay_curve_far_data_txt_name = "decay_curve_far.txt"
        # decay_curve_far_data_file_name = source_dir + decay_curve_far_data_txt_name
        # decay_curve_far_data = np.loadtxt(decay_curve_far_data_file_name, skiprows=16)
        # decay_curve_far_data[:, 0] = decay_curve_far_data[:, 0] + boundary_point
        #
        # decay_curve_full_range_data = np.vstack((decay_curve_data, decay_curve_far_data))
        #
        # shutil.copy(source_dir + txt_name, os.path.dirname(target_folder_name))
        # np.save(os.path.dirname(target_folder_name) + "/decay_curve_full_range_data", decay_curve_full_range_data)
        # print("Successfully finished: ", counter)
        # counter += 1


if __name__ == "__main__":
    main()
    print("Done")
    print("Done")