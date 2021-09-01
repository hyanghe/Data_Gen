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

    save_data_path = './Data_20210302_v01/'

    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    # print("The length of file_paths is: ", len(file_paths))
    file_directories = file_dir[0]
    counter = 1
    params_lst = np.empty((0, 8))
    TTSS_lst = np.empty((0, 1))
    extreme_TTSS = np.empty((0, 9))
    for folder in file_directories:
        source_dir = directory + '/' + folder + '/'
        # os.chdir(temp_working_dir)
        txt_name = "commands_TTSS_template_6.txt"
        source_file_name = source_dir + txt_name
        f = open(source_file_name, 'r')
        lines = f.readlines()
        PowerTotal = int(re.findall("\d+", lines[50].split()[0])[0])
        Die_xy = int(re.findall("\d+", lines[10].split()[0])[0])
        Die_z = int(re.findall("\d+", lines[12].split()[0])[0])
        C_scale = int(re.findall("\d+", lines[32].split()[0])[0])
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        t_Insulator = float(re.findall(match_number, lines[13].split()[0])[0])
        t_Diel = float(re.findall(match_number, lines[14].split()[0])[0])
        K_Diel = float(re.findall(match_number, lines[21].split()[1])[0])
        top_HTC = float(re.findall(match_number, lines[45].split()[0])[0])
        bottom_HTC = float(re.findall(match_number, lines[47].split()[0])[0])
        HTC = top_HTC
        target_folder_name = save_data_path + f"Q_{PowerTotal}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}_C_{C_scale}\\"
        params = np.asarray([PowerTotal, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel, C_scale])
        params_lst = np.vstack((params_lst, params))

        TTSS_txt_name = "TTSS_status.txt"
        TTSS_file_name = source_dir + TTSS_txt_name
        TTSS_f = open(TTSS_file_name, 'r')
        TTSS_lines = TTSS_f.readlines()
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        TTSS_value = float(re.findall(match_number, TTSS_lines[44].split()[1])[0])
        if TTSS_value > 2000:
            extreme_TTSS = np.vstack((extreme_TTSS, np.hstack((params, TTSS_value))))
        TTSS_lst = np.vstack((TTSS_lst, TTSS_value))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    c_scale_lst = params_lst[:, -1] # just as an example array
    ax.scatter(c_scale_lst, TTSS_lst, color='b', marker='*')
    plt.legend(['TTSS distribution'])
    plt.xlabel('C_scale')
    plt.ylabel('TTSS (s)')
    plt.savefig(save_data_path +  f'TTSS distribution')
    print('done generating cscale distribution')
    header = 'power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel, C_scale, TTSS'

    np.savetxt(save_data_path + 'extreme cases.txt', extreme_TTSS, fmt='%.3e', header = header)

    import matplotlib.cm as cm
    # colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'purple']
    markers = ['.', 'x', '^', '*', 'o', 'v', '<', '>', ',']
    legend = ['Total power: 5e2--2e4', 'HTC (log scale): 1e-9--2e-5', 'Die_xy: 4000--30000',\
              'Die_z: 10--200', 't_Diel: 1--10', 't_Insulator: 1e-2--5e-2', 'K_Diel: 138e-5--138e-3',\
              'C_scale: 1e6--30e6']
    colors = cm.rainbow(np.linspace(0, 1, len(legend)))
    val = 0.  # this is the value where you want the data to appear on the y-axis.
    for j in range(params_lst.shape[1]):
        # val += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ar = params_lst[:, j] # just as an example array
        if j == 1:
            ax.plot(np.log10(ar), np.zeros_like(ar) + val, color = colors[j, :], marker =  markers[j])
        else:
            ax.plot(ar, np.zeros_like(ar) + val, color = colors[j, :], marker =  markers[j])
        plt.legend([legend[j]])
        plt.savefig(save_data_path + f'{j}')
        # plt.show()
    print('done')




if __name__ == "__main__":
    main()
    print("Done")
    print("Done")