import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import shutil
import re
path = os.getcwd()

trial = 1
dest_fpath = path + f"/DeepONet_v0{trial}/"
os.makedirs(os.path.dirname(dest_fpath), exist_ok=True, mode=0o777)
ckpt_fpath = path + f"/DeepONet_v0{trial}/Checkpoint/"
os.makedirs(os.path.dirname(ckpt_fpath), exist_ok=True, mode=0o777)
fig_fpath = path + f"/DeepONet_v0{trial}/fig/"
os.makedirs(os.path.dirname(fig_fpath), exist_ok=True, mode=0o777)

os.chmod(path + f"/DeepONet_v0{trial}", 0o777)
for root, dirs, files in os.walk(dest_fpath):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o777)
    for f in files:
        os.chmod(os.path.join(root, f), 0o777)


from zipfile import ZipFile
import os
import numpy as np
import shutil
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_all_file_paths(directory):
    file_paths = []
    file_directories = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
        file_directories.append(directories)
    return file_paths, file_directories


def get_DeepONet_data(train_data, Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel):
    start_time = time.time()
    y_train, s_train = train_data[:, :1], train_data[:, 1:2].reshape(-1, 1)
    ## Try without local minmax scaling
    s_train = s_train - s_train.min()
    max_temp_local = s_train.max()
    min_temp_local = s_train.min()

    power_mag_n = np.log(Power*10**2) * 0.1
    HTC_n = np.log(HTC*10**9) * 0.1
    Die_xy_n = (Die_xy - 4000.0) / (30000.0 - 4000.0)
    Die_z_n = (Die_z - 10.0) / (200.0 - 10.0)
    t_Diel_n = (t_Diel - 1.0) / (10.0 - 1.0)
    t_Insulator_n = (t_Insulator - 0.01) / (0.05 - 0.01)
    K_Diel_n = np.log(K_Diel*10**3) * 0.1

    y_train[:, 0] = y_train[:, 0] / (Die_xy / 2.0)

    u_train = np.asarray([power_mag_n, HTC_n, Die_xy_n, Die_z_n, t_Diel_n, t_Insulator_n, K_Diel_n])
    u_train = np.tile(u_train, reps=(y_train.shape[0], 1))
    ## Scale s_train and s_test
    s_train = (s_train - min_temp_local) / (max_temp_local - min_temp_local)

    end_time = time.time()
    total_time = end_time - start_time
    #     print("Total time consumed is: ", total_time)
    power_test_BC = u_train[0, :].reshape(-1, 1)

    #     print("power_test_BC shape is: ", power_test_BC.shape)
    #     print("power_test_BC max is: ", power_test_BC.max())
    #     print("power_test_BC min is: ", power_test_BC.min())
    return u_train, y_train, s_train


def main():
    # path to folder which needs to be zipped
    directory = './Data_20210118/One_tile_3D_data'
    save_data_path = dest_fpath + 'One_tile_3D_data_npz/'
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    file_directories = file_dir[0]
    counter = 1

    data = 0

    # workbench_u = np.empty((0, 4))
    workbench_u = np.empty((0, 7))
    # workbench_y = np.empty((0, 3))
    workbench_y = np.empty((0, 1))
    workbench_s = np.empty((0, 1))

    for folder in file_directories:
        source_dir = directory + '/' + folder + '/'
        # os.chdir(temp_working_dir)
        txt_name = "commands_tile_heating_HTC_BC_area_effects.txt"
        source_file_name = source_dir + txt_name
        f = open(source_file_name, 'r')
        lines = f.readlines()

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
        target_folder_name = save_data_path + f"Q_{Power}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}\\"


        # tile_size = float(re.findall("\d+", lines[10].split()[0])[0])
        # match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        # top_HTC = float(re.findall(match_number, lines[44].split()[0])[0])
        # bottom_HTC = float(re.findall(match_number, lines[44].split()[0])[0])
        # Power = float(re.findall("\d+", lines[48].split()[0])[0])
        # HTC = top_HTC

        data_file_name = "decay_curve_full_range_data.npy"
        data = np.load(source_dir + data_file_name)
        u, y, s = get_DeepONet_data(data, Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel)
        workbench_u = np.vstack((workbench_u, u))
        workbench_y = np.vstack((workbench_y, y))
        workbench_s = np.vstack((workbench_s, s))
    print(f'workbench_u shape is {workbench_u.shape}')
    print(f'workbench_y shape is {workbench_y.shape}')
    print(f'workbench_s shape is {workbench_s.shape}')
    np.savez_compressed(save_data_path + "workbench_u", workbench_u)
    np.savez_compressed(save_data_path + "workbench_y", workbench_y)
    np.savez_compressed(save_data_path + "workbench_s", workbench_s)


if __name__ == "__main__":
    save_data_path = dest_fpath + 'One_tile_3D_data_npz/'
    main()
    workbench_u_f = np.load(save_data_path + "workbench_u.npz")
    workbench_y_f = np.load(save_data_path + "workbench_y.npz")
    workbench_s_f = np.load(save_data_path + "workbench_s.npz")
    workbench_u = workbench_u_f['arr_0']
    workbench_y = workbench_y_f['arr_0']
    workbench_s = workbench_s_f['arr_0']

    print(f"workbench_y[:, 0] min is {workbench_y[:, 0].min()}")
    print(f"workbench_y[:, 0] max is {workbench_y[:, 0].max()}")
    print(f"workbench_u[:, 0] min is {workbench_u[:, 0].min()}")
    print(f"workbench_u[:, 0] max is {workbench_u[:, 0].max()}")
    print(f"workbench_u[:, 1] min is {workbench_u[:, 1].min()}")
    print(f"workbench_u[:, 1] max is {workbench_u[:, 1].max()}")
    print(f"workbench_u[:, 2] min is {workbench_u[:, 2].min()}")
    print(f"workbench_u[:, 2] max is {workbench_u[:, 2].max()}")
    print(f"workbench_u[:, 3] min is {workbench_u[:, 3].min()}")
    print(f"workbench_u[:, 3] max is {workbench_u[:, 3].max()}")
    print(f"workbench_u[:, 4] min is {workbench_u[:, 4].min()}")
    print(f"workbench_u[:, 4] max is {workbench_u[:, 4].max()}")
    print(f"workbench_u[:, 5] min is {workbench_u[:, 5].min()}")
    print(f"workbench_u[:, 5] max is {workbench_u[:, 5].max()}")
    print(f"workbench_u[:, 6] min is {workbench_u[:, 6].min()}")
    print(f"workbench_u[:, 6] max is {workbench_u[:, 6].max()}")
    print(f"workbench_s min is {workbench_s.min()}")
    print(f"workbench_s max is {workbench_s.max()}")