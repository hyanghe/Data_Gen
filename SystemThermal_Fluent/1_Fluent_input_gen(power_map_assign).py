import numpy as np
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import glob
import shutil
# np.random.seed(123)


# cur_path = os.getcwd()
# shutil.rmtree(cur_path)
# shutil.copy(os.path.get_dir(cur_path) + 'run_fluent_new_noudf_v06.jou', cur_path)

def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6  # .reshape(-1, *blck)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def block(data, small_blocks, dimX, dimY):
    data_shape = data.reshape(dimX[0], dimX[1], dimX[2])
    dummy1 = cutup(data_shape, (dimY[0], dimY[1], dimY[2]), small_blocks)
    dummy2 = dummy1.reshape(dummy1.shape[0] * dummy1.shape[1] * dummy1.shape[2], dimY[0] * dimY[1] * dimY[2])

    return dummy2

num_maps = 4 # total cases to generate
max_gaussians = 200 # maximum number of gaussians in power map distribution
net_power = 1 # Net power Watts
smoothing_iter = 20

# These are coordinates of the mesh nodes we want to specify the power map on
# Coordinates are normalized between 0.0 and 1.0
nx = 200
ny = 20
# grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx))
grid_x, grid_y = np.meshgrid(np.linspace(0.0, 0.016, nx), np.linspace(0.0, 0.016, nx))
points = np.concatenate((grid_x.flatten().reshape(-1,1), grid_y.flatten().reshape(-1, 1)), 1)
power = np.zeros((nx, nx, 1))

small_blocks = (ny, ny, 1)
domain_size = [nx, nx, 1]
subdomain_size = [ny, ny, 1]
pwr_block = block(power, small_blocks, domain_size, subdomain_size)
pwr_block = pwr_block.reshape(pwr_block.shape[0], ny, ny, 1)

dest_dir = os.getcwd() + '\Fluent_inputs'
os.makedirs(dest_dir, mode=0o777, exist_ok=True)
# seed_dir = os.getcwd() + '\case1'
seed_dir = os.getcwd() + '\\NaturalConvection_data_gen'

case_seeds = glob.glob(f'{seed_dir}\*.cas')
c_seeds = glob.glob(f'{seed_dir}\*.c')
counter = 0
for j in range(num_maps):
    for case_seed in case_seeds:
        # power_scale = np.random.randint(low=5000, high=120000)
        # power_scale = np.random.randint(low=0, high=5*10**7)
        power_scale = np.random.randint(low=0, high=10 * 10 ** 7)
    # for case_seed in case_seeds[:1]:
        src_case_seed = case_seed
        os.chmod(src_case_seed, mode=0o777)
        des_dir = dest_dir + '\\' + os.path.split(case_seed)[-1].split('.')[0] + f'_power_{power_scale}'
        os.makedirs(des_dir, mode=0o777, exist_ok=True)

        shutil.copy(src_case_seed, des_dir+'\\trial001.cas')

        src_c_seed = c_seeds[0]
        shutil.copy(src_c_seed, des_dir)
        for k in range(pwr_block.shape[0]):
            a = np.random.exponential(0.01)
            pwr_block[k, :] = a

        pwr_block1 = uncubify(pwr_block, domain_size)
        kernel = (np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]))
        kernel = kernel/np.sum(kernel)
        pwr_block1 = pwr_block1[:, :, 0]
        for p in range(smoothing_iter):
            pwr_block1 = ndimage.convolve(pwr_block1, kernel, mode='nearest')
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 60000000
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 625000000  ## total power: 160000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 312500000  ## total power: 80000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 156250000  ## total power: 40000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 78125000  ## total power: 20000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 15625000  ## total power: 4000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 31250000  ## total power: 8000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 46875000  ## total power: 12000W, x: 0.016 m, y: 0.016 m
        pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * power_scale  ## total power: 16000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 7812500  ## total power: 2000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 3906250  ## total power: 1000W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 1953125  ## total power: 500W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 781250 ## total power: 200W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 78125.0  ## total power: 20W, x: 0.016 m, y: 0.016 m
        # pwr_block1 = pwr_block1 * net_power / np.sum(pwr_block1) * 7812.50  ## total power: 2W, x: 0.016 m, y: 0.016 m

        # target_file_name = des_dir + '\\' + os.path.split(case_seed)[-1]
        target_file_name = des_dir + '\\trial001.cas'
        f_case = open(target_file_name,'r+')
        lines = f_case.readlines()
        flag = 0
        for line_idx, line in enumerate(lines[::-1]):
            if '(profile000' in line:
                flag = len(lines) - line_idx
                break
        lines[flag - 1] = lines[flag - 1][:-3] + f'{pwr_block1.shape[0] * pwr_block1.shape[1]})\n'
        lines[flag:] = ''
        f_case.close()
        f_case = open(target_file_name, 'w')
        f_case.writelines(lines)
        f_case.close()
        # f_case.close()
        power_values = pwr_block1.reshape((-1, 1))
        np.save(des_dir + '/power_map', power_values)
        # with open(des_dir + '\\' + os.path.split(case_seed)[-1], 'a') as f_case:
        with open(des_dir + '\\' + 'trial001.cas', 'a') as f_case:
            f_case.write('\n(x')
            x_coords = grid_x.reshape((-1, 1))
            for x_coord in x_coords:
                f_case.write(f'\n{x_coord[0]}')
            f_case.write('\n)')
            f_case.write('\n')

            f_case.write('\n(y')
            y_coords = grid_y.reshape((-1, 1))
            for y_coord in y_coords:
                f_case.write(f'\n{y_coord[0]}')
            f_case.write('\n)')
            f_case.write('\n')

            f_case.writelines('\n(z')
            for _ in range(len(x_coords)):
                f_case.writelines(f'\n{0.001622}')
            f_case.write('\n)')
            f_case.write('\n')

            f_case.writelines('\n(q')

            for power_value in power_values:
                f_case.write(f'\n{power_value[0]}')
            f_case.write('\n)')
            f_case.write('\n')
            f_case.write('\n)')
            f_case.close()
        if counter == 0:
            journal_file = os.getcwd() + '\Fluent_inputs' + '\\' + 'run_fluent_new_noudf_v07.jou'
            with open(journal_file, 'r') as journal:
                lines_j = journal.readlines()
                lines_new = lines_j
                case_name = os.path.split(case_seed)[-1].split('.')[0] + f'_power_{power_scale}'
                # search_key = os.path.split(case_seed)[-1].split('.')[0][:8]
                search_key = 'trial001_profile_0'
                idx_l2 = lines_new[2].index(search_key)
                lines_new[2] = lines_new[2][:idx_l2] + 'Fluent_inputs/' + case_name  + lines_new[2][idx_l2 + len(search_key):]
                idx_l5 = lines_new[5].index(search_key)
                lines_new[5] = lines_new[5][:idx_l5] + 'Fluent_inputs/' + case_name + lines_new[5][idx_l5 + len(search_key):]
                # idx_l6 = lines_new[6].index(search_key)
                # lines_new[6] = lines_new[6][:idx_l6] + 'Fluent_inputs/' + case_name + lines_new[6][idx_l6 + len(search_key):]
                idx_l26 = lines_new[26].index(search_key)
                lines_new[26] = lines_new[26][:idx_l26] + 'Fluent_inputs/' + case_name + lines_new[26][idx_l26 + len(search_key):]
            with open(journal_file, 'w') as journal:
                journal.writelines(lines_new)
            first_case_name = case_name
        if counter > 0:
            with open(os.getcwd() + '\Fluent_inputs' + '\\' + 'run_fluent_new_noudf_v07.jou', 'r') as journal:
                lines_j = journal.readlines()
                lines_new = lines_j
                case_name = os.path.split(case_seed)[-1].split('.')[0] + f'_power_{power_scale}'
                search_key = os.path.split(case_seed)[-1].split('.')[0][:8]
                idx_l2 = lines_new[2].index(search_key)
                lines_new[2] = lines_new[2][:idx_l2]  + case_name  + lines_new[2][idx_l2+len(first_case_name):]
                idx_l5 = lines_new[5].index(search_key)
                lines_new[5] = lines_new[5][:idx_l5]  + case_name + lines_new[5][idx_l5 + len(first_case_name):]
                # idx_l6 = lines_new[6].index(search_key)
                # lines_new[6] = lines_new[6][:idx_l6]  + case_name + lines_new[6][idx_l6 + len(first_case_name):]
                # idx_l27 = lines_new[27].index(search_key)
                # lines_new[27] = lines_new[27][:idx_l27]  + case_name + lines_new[27][idx_l27 + len(first_case_name):]
                idx_l26 = lines_new[26].index(search_key)
                lines_new[26] = lines_new[26][:idx_l26]  + case_name + lines_new[26][idx_l26 + len(first_case_name):]
            with open(os.getcwd() + '\Fluent_inputs' + '\\' + 'run_fluent_new_noudf_v07.jou', 'a') as jour:
                jour.write('\n')
                # for line_new in lines_new[1:28]:
                for line_new in lines_new[1:27]:
                    jour.write(line_new)
                # print('h')
            # f_case = open(des_dir + '\\' + os.path.split(case_seed)[-1], 'w')
            # f_case.writelines(lines)
            # f_case.close()
        # print(np.sum(pwr_block1))
        print('Max heat flux is: ', pwr_block1.max())
        counter += 1
            # cmap = plt.cm.jet
            # imgplot = plt.imshow(pwr_block1.reshape(nx, nx), cmap=cmap)
            # plt.savefig("case_%d.png" % (j))

