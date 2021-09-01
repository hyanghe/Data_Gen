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
from scipy.interpolate import griddata

def get_all_file_paths(directory):
    file_paths = []
    file_directories = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
        file_directories.append(directories)
    return file_paths, file_directories


def get_DeepONet_data(train_data, Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel, TTSS_value, C_scale):
    start_time = time.time()
    y_train, s_train = train_data[:, :1], train_data[:, 1:2].reshape(-1, 1)
    ## Try without local minmax scaling
    y_train[:, 0] = (y_train[:, 0] - y_train[:, 0].min()) / (y_train[:, 0].max() - y_train[:, 0].min())

    s_train = s_train - s_train.min()
    max_temp_local = s_train.max()
    min_temp_local = s_train.min()
    #     print('max_temp_local is: ', max_temp_local)
    #     print('min_temp_local is: ', min_temp_local)
    ## Scale s_train and s_test
    s_train = (s_train - min_temp_local) / (max_temp_local - min_temp_local)

    y_train_grid = np.linspace(0, 1, 101, endpoint=True)
    s_train_interpolate = griddata(y_train, s_train, y_train_grid)
    y_train = y_train_grid[:, None]
    s_train = s_train_interpolate

    #     if np.isnan(np.sum(s_train)):
    #         print('s_train is: ', s_train)
    #         print('Power is: ', Power)
    #         print('HTC is: ', HTC)
    #         print('Die_xy is: ', Die_xy)
    #         print('C_scale is: ', C_scale)
    #     print('s_train min is: ', s_train.min())
    #     print('s_train max is: ', s_train.max())

    # power_mag_n = np.log(Power*10**2) * 0.1
    power_mag_n = np.log(Power * 2 * 10 ** -3) * 0.25
    HTC_n = np.log(HTC * 10 ** 9) * 0.1
    Die_xy_n = (Die_xy - 4000.0) / (30000.0 - 4000.0)
    Die_z_n = (Die_z - 10.0) / (200.0 - 10.0)
    t_Diel_n = (t_Diel - 1.0) / (10.0 - 1.0)
    t_Insulator_n = (t_Insulator - 0.01) / (0.05 - 0.01)
    K_Diel_n = np.log(K_Diel * 10 ** 3) * 0.1
    TTSS_value_n = TTSS_value / 2000.0

    u_train = np.asarray([power_mag_n, HTC_n, Die_xy_n, Die_z_n, t_Diel_n, t_Insulator_n, K_Diel_n, TTSS_value_n])
    u_train = np.tile(u_train, reps=(y_train.shape[0], 1))

    end_time = time.time()
    total_time = end_time - start_time
    #     print("Total time consumed is: ", total_time)
    # power_test_BC = u_train[0, :].reshape(-1, 1)

    #     print("power_test_BC shape is: ", power_test_BC.shape)
    #     print("power_test_BC max is: ", power_test_BC.max())
    #     print("power_test_BC min is: ", power_test_BC.min())
    return u_train, y_train, s_train


def main():
    # path to folder which needs to be zipped
    directory = './DeepONetData_20210224_v01_10_2000/One_tile_3D_data'
    save_data_path = dest_fpath + 'One_tile_3D_data_npz/'
    os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

    file_paths, file_dir = get_all_file_paths(directory)
    file_directories = file_dir[0]
    counter = 1

    data = 0

    # workbench_u = np.empty((0, 4))
    workbench_u = np.empty((0, 8))
    # workbench_y = np.empty((0, 3))
    workbench_y = np.empty((0, 1))
    workbench_s = np.empty((0, 1))
    batch_id = np.empty((0, 1), dtype=int)
    batch_label = np.empty((0, 1), dtype=int)
    cnt = 0
    for folder in file_directories:
        source_dir = directory + '/' + folder + '/'
        # os.chdir(temp_working_dir)
        txt_name = "commands_TTSS_template_5.txt"
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
        HTC = top_HTC
        target_folder_name = save_data_path + f"Q_{PowerTotal}_HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}_Kd_{K_Diel}_C_{C_scale}\\"

        TTSS_txt_name = "TTSS_status.txt"
        TTSS_file_name = source_dir + TTSS_txt_name
        TTSS_f = open(TTSS_file_name, 'r')
        TTSS_lines = TTSS_f.readlines()
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        TTSS_value = float(re.findall(match_number, TTSS_lines[42].split()[1])[0])

        data_file_name = "TTSS_data.npy"
        data = np.load(source_dir + data_file_name)
        idx = np.where(data[:, 0] == TTSS_value)[0][0]
        filtered_data = data[:idx + 1, :]
        u, y, s = get_DeepONet_data(filtered_data, PowerTotal, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel,
                                    TTSS_value, C_scale)

        workbench_u = np.vstack((workbench_u, u))
        workbench_y = np.vstack((workbench_y, y))
        workbench_s = np.vstack((workbench_s, s))
        batch_label = np.vstack((batch_label, np.ones((u.shape[0], 1)) * cnt))
        batch_id = np.vstack((batch_id, u.shape[0]))
        cnt += 1
    print(f'workbench_u shape is {workbench_u.shape}')
    print(f'workbench_y shape is {workbench_y.shape}')
    print(f'workbench_s shape is {workbench_s.shape}')
    print(f'batch_id shape is {batch_id.shape}')
    np.savez_compressed(save_data_path + "workbench_u", workbench_u)
    np.savez_compressed(save_data_path + "workbench_y", workbench_y)
    np.savez_compressed(save_data_path + "workbench_s", workbench_s)
    np.savez_compressed(save_data_path + "batch_id", batch_id)
    np.savez_compressed(save_data_path + "batch_label", batch_label)


if __name__ == "__main__":
    save_data_path = dest_fpath + 'One_tile_3D_data_npz/'
    main()
    workbench_u_f = np.load(save_data_path + "workbench_u.npz")
    workbench_y_f = np.load(save_data_path + "workbench_y.npz")
    workbench_s_f = np.load(save_data_path + "workbench_s.npz")
    batch_id_f = np.load(save_data_path + "batch_id.npz")
    batch_label_f = np.load(save_data_path + "batch_label.npz")
    workbench_u = workbench_u_f['arr_0']
    workbench_y = workbench_y_f['arr_0']
    workbench_s = workbench_s_f['arr_0']
    batch_id = batch_id_f['arr_0']
    batch_label = batch_label_f['arr_0']

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
    print(f"workbench_u[:, 7] min is {workbench_u[:, 7].min()}")
    print(f"workbench_u[:, 7] max is {workbench_u[:, 7].max()}")
    print(f"batch_id min is {batch_id.min()}")
    print(f"batch_id max is {batch_id.max()}")
    print(f"batch_label min is {batch_label.min()}")
    print(f"batch_label max is {batch_label.max()}")
    print(f"workbench_s min is {workbench_s.min()}")
    print(f"workbench_s max is {workbench_s.max()}")

save_data_path = dest_fpath + 'One_tile_3D_data_npz/'
workbench_u_f = np.load(save_data_path + "workbench_u.npz")
workbench_y_f = np.load(save_data_path + "workbench_y.npz")
workbench_s_f = np.load(save_data_path + "workbench_s.npz")
batch_id_f = np.load(save_data_path + "batch_id.npz")
workbench_u = workbench_u_f['arr_0']
workbench_y = workbench_y_f['arr_0']
workbench_s = workbench_s_f['arr_0']
batch_id = batch_id_f['arr_0']
batch_label = batch_label_f['arr_0']

start_idx = [0]
end_idx = [batch_id[0][0]]
for i, ids in enumerate(batch_id[:-1]):
    start_idx.append(start_idx[-1] + ids[0])
    end_idx.append(end_idx[-1] + batch_id[i + 1, :][0])
indexer = tuple(map(slice,start_idx,end_idx))


total_num_data = len(indexer)
print("total_num_data is: ", total_num_data)
train_idx = np.random.choice(total_num_data, size=total_num_data * 19 // 20, replace=False)
test_idx = np.setxor1d(np.arange(total_num_data), train_idx)

# train_idx = np.random.choice(total_num_data, size = total_num_data, replace = False)
# test_idx = train_idx

# train_idx = np.append(np.arange(25, 50), np.arange(175, 225))
# test_idx = train_idx

u_train = np.empty((0, workbench_u.shape[1]))
y_train = np.empty((0, workbench_y.shape[1]))
s_train = np.empty((0, workbench_s.shape[1]))
sdf_train = np.empty((0, workbench_y.shape[1]))
start_idx_tr = [0]
end_idx_tr = [workbench_u[indexer[train_idx[0]]].shape[0]]
# start_idx_tr = []
# end_idx_tr = []
for idx in train_idx:
    t_u = workbench_u[indexer[idx]]
    t_y = workbench_y[indexer[idx]]
    t_s = workbench_s[indexer[idx]]

    start_idx_tr.append(start_idx_tr[-1] + t_u.shape[0])

    sdf = 1 / (1 + np.exp(10 * t_y - 4))

    u_train = np.vstack((u_train, t_u))
    y_train = np.vstack((y_train, t_y))
    s_train = np.vstack((s_train, t_s))
    sdf_train = np.vstack((sdf_train, sdf))
print("The shape of u_train is: ", u_train.shape)
print("The shape of y_train is: ", y_train.shape)
print("The shape of s_train is: ", s_train.shape)
print("The shape of sdf_train is: ", sdf_train.shape)

u_test = np.empty((0, workbench_u.shape[1]))
y_test = np.empty((0, workbench_y.shape[1]))
s_test = np.empty((0, workbench_s.shape[1]))

start_idx_ge = [0]
end_idx_ge = [workbench_u[indexer[test_idx[0]]].shape[0]]
for idx in test_idx:
    t_u = workbench_u[indexer[idx]]
    t_y = workbench_y[indexer[idx]]
    t_s = workbench_s[indexer[idx]]
    start_idx_ge.append(start_idx_ge[-1] + t_u.shape[0])
    u_test = np.vstack((u_test, t_u))
    y_test = np.vstack((y_test, t_y))
    s_test = np.vstack((s_test, t_s))
print("The shape of u_test is: ", u_test.shape)
print("The shape of y_test is: ", y_test.shape)
print("The shape of s_test is: ", s_test.shape)

power_test_BC = u_train[0, :].reshape(-1, 1)
# print("The shape of temp_scaling_factor_train is: ", temp_scaling_factor_train.shape)
# print("The shape of temp_scaling_factor_test is: ", temp_scaling_factor_test.shape)

start_idx_tr = start_idx_tr[:-1]
end_idx_tr = start_idx_tr[1:]

indexer_tr = tuple(map(slice, start_idx_tr, end_idx_tr))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('sdf sample')
plt.plot(y_train[indexer_tr[0]], sdf_train[indexer_tr[0]])

start_idx_ge = start_idx_ge[:-1]
end_idx_ge = start_idx_ge[1:]
indexer_ge = tuple(map(slice, start_idx_ge, end_idx_ge))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('TTSS curve sample')
plt.plot(y_test[indexer_ge[0]], s_test[indexer_ge[0]])
