from zipfile import ZipFile
# from SALib.sample import sobol_sequence
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
    directory = './SolverGeneratedData'

    save_data_path = './Extracted_Data/One_tile_3D_data/'
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    # print("The length of file_paths is: ", len(file_paths))
    file_directories = file_dir[0]
    counter = 1
    for folder in file_directories:
        if not os.path.isfile(f'{directory}\\{folder}' + '\\decay_curve.txt'):
            src_dir_not_completed = f'{directory}\\{folder}'
            dst_dir_not_completed = f'.\\NotCompleted\\{folder}'
            shutil.move(src_dir_not_completed, dst_dir_not_completed)
            print(f'moved {src_dir_not_completed} to {dst_dir_not_completed}')
            continue

        source_dir = directory + '/' + folder + '/'
        # os.chdir(temp_working_dir)
        txt_name = "commands_tile_heating_HTC_BC_area_effects_Haiyang_1.txt"
        source_file_name = source_dir + txt_name
        f = open(source_file_name, 'r')
        lines = f.readlines()
        # tile_size = float(re.findall("\d+", lines[10].split()[0])[0])
        Die_xy = int(re.findall("\d+", lines[10].split()[0])[0])
        Die_z = int(re.findall("\d+", lines[12].split()[0])[0])

        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        t_Insulator = float(re.findall(match_number, lines[13].split()[0])[0])
        t_Diel = float(re.findall(match_number, lines[14].split()[0])[0])
        tile_size = int(re.findall(match_number, lines[15].split()[0])[0])

        K_Diel = float(re.findall(match_number, lines[33].split()[1])[0])


        top_HTC = float(re.findall(match_number, lines[49].split()[0])[0])
        bottom_HTC = float(re.findall(match_number, lines[51].split()[0])[0])
        Power = float(re.findall(match_number, lines[55].split()[0])[0])
        HTC = top_HTC
        # target_folder_name = save_data_path + f"TILE_SIZE_{int(tile_size)}_Q_{int(Power)}_HTC_{HTC}/"
        # target_folder_name = save_data_path + f"Q_{Power}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}\\"
        target_folder_name = save_data_path + f"Q_{Power}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}_Tile_{tile_size}\\"


        decay_curve_data_txt_name = "decay_curve.txt"
        decay_curve_data_file_name = source_dir + decay_curve_data_txt_name

        decay_f = open(decay_curve_data_file_name, 'r').readlines()

        start = 1002
        end = 1014
        del decay_f[start:end]
        for i in range(1, len(decay_f)//999):
            start = start + 999 - 13
            end = end + 999 - 13
            del decay_f[start:end+1]
        del decay_f[:18]

        if not os.path.exists(target_folder_name):
            os.makedirs(target_folder_name, exist_ok=True, mode=0o777)

        decay_curve_data_file_name_updated = os.path.dirname(target_folder_name) + "/decay_curve_filtered.txt"
        decay_f_v01 = open(decay_curve_data_file_name_updated, "w+")
        for line in decay_f:
            decay_f_v01.write(line)
        decay_f_v01.close()
        decay_curve_data = np.loadtxt(decay_curve_data_file_name_updated)
        boundary_point = decay_curve_data[-1][0]

        decay_curve_far_data_txt_name = "decay_curve_far.txt"
        decay_curve_far_data_file_name = source_dir + decay_curve_far_data_txt_name
        decay_curve_far_data = np.loadtxt(decay_curve_far_data_file_name, skiprows=19) ##Delete the first point at decay_curve_far
        decay_curve_far_data[:, 0] = decay_curve_far_data[:, 0] + boundary_point

        decay_curve_full_range_data = np.vstack((decay_curve_data, decay_curve_far_data))

        shutil.copy(source_dir + txt_name, os.path.dirname(target_folder_name))
        np.save(os.path.dirname(target_folder_name) + "/decay_curve_full_range_data", decay_curve_full_range_data)
        print("Successfully finished: ", counter)
        counter += 1


if __name__ == "__main__":
    main()
    print("Done")
    print("Done")