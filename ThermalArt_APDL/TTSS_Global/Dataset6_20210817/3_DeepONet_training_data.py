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
    directory = './SolverGeneratedData_v02'

    save_data_path = './DeepONetData_20210302_v01_10_2000/One_tile_3D_data/'
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    # print("The length of file_paths is: ", len(file_paths))
    file_directories = file_dir[0]
    counter = 1
    for folder in file_directories:
        source_dir = directory + '/' + folder + '/'
        # os.chdir(temp_working_dir)
        txt_name = "commands_TTSS_template_6.txt"
        source_file_name = source_dir + txt_name
        f = open(source_file_name, 'r')
        lines = f.readlines()

        PowerTotal = int(re.findall("\d+", lines[45].split()[0])[0])
        Die_xy = int(re.findall("\d+", lines[8].split()[0])[0])
        Die_z = int(re.findall("\d+", lines[10].split()[0])[0])
        C_scale = int(re.findall("\d+", lines[30].split()[0])[0])
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        t_Insulator = float(re.findall(match_number, lines[11].split()[0])[0])
        t_Diel = float(re.findall(match_number, lines[12].split()[0])[0])
        K_Diel = float(re.findall(match_number, lines[19].split()[1])[0])
        top_HTC = float(re.findall(match_number, lines[40].split()[0])[0])
        bottom_HTC = float(re.findall(match_number, lines[42].split()[0])[0])
        # Power = float(re.findall(match_number, lines[48].split()[0])[0])
        HTC = top_HTC
        # target_folder_name = save_data_path + f"TILE_SIZE_{int(tile_size)}_Q_{int(Power)}_HTC_{HTC}/"
        # target_folder_name = save_data_path + f"Q_{Power}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}\\"
        target_folder_name = save_data_path + f"Q_{PowerTotal}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}_C_{C_scale}\\"

        TTSS_data_txt_name = "TTSS_curve.txt"
        TTSS_data_file_name = source_dir + TTSS_data_txt_name
        TTSS_f = open(TTSS_data_file_name, 'r').readlines()


        TTSS_status_txt_name = "TTSS_status.txt"
        TTSS_status_file_name = source_dir + TTSS_status_txt_name
        TTSS_status_f = open(TTSS_status_file_name, 'r')
        TTSS_status_lines = TTSS_status_f.readlines()
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        TTSS_status_value = float(re.findall(match_number, TTSS_status_lines[42].split()[1])[0])
        if TTSS_status_value < 2000 and TTSS_status_value > 10:
            if not os.path.exists(target_folder_name):
                os.makedirs(target_folder_name, exist_ok=True, mode=0o777)

            shutil.copy(source_dir + txt_name, os.path.dirname(target_folder_name))
            shutil.copy(source_dir + TTSS_status_txt_name, os.path.dirname(target_folder_name))

            TTSS_data_file_name_updated = os.path.dirname(target_folder_name) + "/TTSS_curve.txt"
            TTSS_f_v01 = open(TTSS_data_file_name_updated, "w+")
            for line in TTSS_f:
                TTSS_f_v01.write(line)
            TTSS_f_v01.close()
            TTSS_curve_data_np = np.loadtxt(TTSS_data_file_name_updated, skiprows=1)



            np.save(os.path.dirname(target_folder_name) + "/TTSS_data", TTSS_curve_data_np)

            # # np.save(os.path.dirname(target_folder_name) + "/TTSS_full_range_data", decay_curve_full_range_data)
            # np.save(os.path.dirname(target_folder_name) + "/TTSS_curve_data", TTSS_curve_data)
            print("Successfully finished: ", counter)
            counter += 1


if __name__ == "__main__":
    main()
    print("Done")
    print("Done")