import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt



cur_path = os.getcwd()
data_dir = cur_path + '\\Fluent_outputs\\ML_Solver_data_96\\block_data'
data_files = glob.glob(data_dir + '\\*.pickle')

all_case_dir = cur_path + '\\Fluent_outputs\\Fluent_outputs_v04_96\\'
case_files = os.listdir(all_case_dir)


# cur_path = os.getcwd()
# data_dir = cur_path + '\\Fluent_outputs\\ML_Solver_data_np_cell\\block_data'
# data_files = glob.glob(data_dir + '\\*.pickle')

# all_case_dir = cur_path + '\\Fluent_outputs\\Fluent_outputs_v02_382_np\\'
# case_files = os.listdir(all_case_dir)


power_maps = np.empty((0, 200, 200))
temp_mins = np.empty((0, 1))
params = np.empty((0, 2))
cnt = 0
for case_file, data_file in zip(case_files, data_files):
    if case_file == data_file.split('\\')[-1].split('.')[0][13:]:
        print('cnt=', cnt)
        cnt += 1
        power_map = np.load(glob.glob(all_case_dir+case_file + '\\' + "*.npy")[0])
        power_maps = np.concatenate((power_maps, power_map.reshape((1, 200, 200))), axis=0)

        with open(data_file, "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        temp_min = np.asarray([[data['solution']['temperature'][0].min()]])
        temp_mins = np.concatenate((temp_mins, temp_min), axis=0)

        param_str = case_file.split('_')
        AmbT = int(param_str[2])
        FinCount = int(param_str[4])
        if param_str[6] == 'Off':
            FinCount = 0
        Total_power = int(param_str[-1])

        param = np.asarray([[AmbT, FinCount]])
        params = np.concatenate((params, param), axis=0)

T_min_predictor_data_dir = cur_path + '\\T_min_predictor\\data\\'
os.makedirs(T_min_predictor_data_dir, mode=0o777,exist_ok=True)
np.savez_compressed(T_min_predictor_data_dir + 'power_maps_362.npz', power_maps)
np.savez_compressed(T_min_predictor_data_dir + 'temp_mins_362.npz', temp_mins)
np.savez_compressed(T_min_predictor_data_dir + 'params_362.npz', params)

fig = plt.figure()
ax = fig.add_subplot(111)
true = temp_mins
x = np.arange(true.shape[0])
plt.scatter(x, true, c = 'b')
ax.legend(['Min temp'])
ax.set_xlabel('case number')
ax.set_ylabel('Temperature')
T_min_predictor_folder = '.\\T_min_predictor\\'
plt.savefig(T_min_predictor_folder + 'Min_pred_all_362')
