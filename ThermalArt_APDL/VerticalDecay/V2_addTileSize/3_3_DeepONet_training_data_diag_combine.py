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

    save_data_path = './Data_20210625/One_tile_3D_data/'
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    # print("The length of file_paths is: ", len(file_paths))
    file_directories = file_dir[0]
    counter = 1
    for folder in file_directories:
        source_dir = save_data_path + '/' + folder + '/'
        decay_curve_full = np.load(source_dir + 'decay_curve_full_range_data.npy')
        decay_curve_full_diag = np.load(source_dir + 'decay_curve_full_range_data_diag.npy')
        decay_curve_end_pt = decay_curve_full[-1, 0]

        idx_interface, location_interface = min(enumerate(decay_curve_full_diag[:, 0]), key = lambda x: abs(x[1] - decay_curve_end_pt))
        temp_diff_interface = decay_curve_full[-1, 1] - decay_curve_full_diag[idx_interface, 1]
        decay_curve_full_diag_tails_loc = decay_curve_full_diag[idx_interface:, 0:1]
        decay_curve_full_diag_tails_temp = decay_curve_full_diag[idx_interface:, 1:2] + temp_diff_interface
        decay_curve_full_diag_tails = np.hstack((decay_curve_full_diag_tails_loc, decay_curve_full_diag_tails_temp))

        x_coord_interp = np.linspace(decay_curve_full_diag_tails_loc.min(),\
                                   decay_curve_full_diag_tails_loc.max(),\
                                   100)
        y_interp = np.interp(x_coord_interp,\
                             decay_curve_full_diag_tails_loc.flatten(),\
                             decay_curve_full_diag_tails_temp.flatten())
        x_coord_y_interp = np.hstack((x_coord_interp[:, None], y_interp[:, None]))
        decay_curve_full_diag_full = np.concatenate((decay_curve_full, x_coord_y_interp), axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(decay_curve_full[:, 0], decay_curve_full[:, 1], 'bo-')
        ax.plot(decay_curve_full_diag[:, 0], decay_curve_full_diag[:, 1], 'r*-')
        ax.plot(decay_curve_full_diag_full[:, 0], decay_curve_full_diag_full[:, 1], 'k<-')
        ax.legend(['Decay curve', 'Decay curve diagonal', 'Decay curve plus digonal shift'])

        np.save(source_dir + 'decay_curve_full_range_diag_full_range_data.npy', decay_curve_full_diag_full)
        print('counter: ', counter)
        counter += 1
        print('done')


if __name__ == "__main__":
    main()
    print("Done")
    print("Done")