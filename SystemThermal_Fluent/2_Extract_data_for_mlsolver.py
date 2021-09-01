import numpy as np
import matplotlib
matplotlib.use('agg')
import os, re, time
import pandas as pd
from src.ml_solver.Processor import Processor
from src.geometry_encoder import levelset
from src.utility import interpolation, binning, write_to_file
from src.utility.plotting import MatplotlibPlotter
import glob
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# from case_setup import case_setup
import logging

data_save_folder_name = "ML_Solver_data_96"
class FluentDataPreprocessor():
    def __init__(self, path, rank, files_per_proc, case_setup, testing):

        self.path = path
        self.rank = rank
        self.files_per_proc = files_per_proc
        self.ml_proc = Processor(case_setup["list_var_names"], case_setup["domain_size"], case_setup["subdomain_size"])
        self.tot_subdomains = self.ml_proc.tot_patches
        self.domain_size = case_setup["domain_size"]
        self.subdomain_size = case_setup["subdomain_size"]
        self.sampling_indices_list = case_setup["sampling_indices"]
        self.var_names = case_setup["list_var_names"]
        self.num_var = case_setup["num_of_vars"]
        self.testing = testing
        self.filepaths = case_setup["filepaths"]
        self.sampling = case_setup["preprocessor_sampling"]
        self.plot_interp = case_setup["plot_interpolated"]





        if not case_setup:
            print("Specify boundary conditions, source or other conditions used in simulation")
            exit()
        else:
            if "sourceterm_global" in case_setup["Simulation_features"] or "sourceterm_local" in case_setup[
                "Simulation_features"]:
                # self.source_size = case_setup["sourceterm_size"]
                # self.source = case_setup["sourceterm"]
                # self.source = self.source[self.rank * self.files_per_proc:(self.rank + 1) * self.files_per_proc, :]
                # self.source = self.source.reshape(self.files_per_proc, self.source_size)
                self.source = []

            if "boundary_condition" in case_setup["Simulation_features"]:
                self.bcs = case_setup["boundary_condition"]
                # self.bcs = self.bcs[self.rank * self.files_per_proc:(self.rank + 1) * self.files_per_proc, :]

        self.sim_features = case_setup["Simulation_features"]
        # Fluent data consists of flow solutions and geometry metrics on each cell.
        # The data files for each case are in ASCII format and the columns
        # The columns are ordered as:
        # cell number, x-coordinate, y-coordinate, z-coordinate, cell zone, temperature, x-velocity, y-velocity, z-velocity, pressure

    def get_filenames(self, path, rank, files_per_proc):

        filenames_g = []
        for filename in os.listdir(path):
            filenames_g.append(filename)

        #filenames_g.sort(key=lambda f: int(re.sub('\D', '', f)))

        # filenames = filenames_g[rank * files_per_proc:(rank + 1) * files_per_proc]

        return filenames_g

    def initialize_dictionaries(self):

        blocked_dict = {}
        blocked_dict['solution'] = {}
        for j in self.var_names:
            blocked_dict['solution'][j] = []

        if "geometry" in self.sim_features or "geometry_global" in self.sim_features:
            blocked_dict['geometry'] = []
        if "sourceterm_global" in self.sim_features or "sourceterm_local" in self.sim_features:
            blocked_dict['sourceterm'] = []
        if "boundary_condition" in self.sim_features:
            blocked_dict['boundary_condition'] = []

        point_dict = {}
        point_dict['solution'] = {}
        for j in self.var_names:
            point_dict['solution'][j] = []

        if "geometry" in self.sim_features:
            point_dict['geometry'] = []
        if "sourceterm_global" in self.sim_features or "sourceterm_local" in self.sim_features:
            point_dict['sourceterm'] = []
        if "boundary_condition" in self.sim_features:
            point_dict['boundary_condition'] = []
        point_dict['points'] = {}
        point_dict['points']['coordinates'] = []

        for j in self.var_names:
            point_dict['points'][j] = []

        return blocked_dict, point_dict

    def read_fluent_data(self,):

        filenames = self.get_filenames(self.path, self.rank, self.files_per_proc)
        count = 0
        lv = levelset.LevelSet()
        interp = interpolation.Interpolation()
        binner = binning.Binning()
        write = write_to_file.Writer()

        count = -1
        cwd_path = os.getcwd()
        folderpath = cwd_path + f"/Fluent_outputs/{data_save_folder_name}/"
        filepath = folderpath+"info" + str(self.rank) + ".txt"

        if self.testing:
            filenames_all = [filenames[self.files_per_proc-1]]
        else:
            filenames_all = filenames


        x_coord_max = -100000000
        x_coord_min = 100000000
        y_coord_max = -100000000
        y_coord_min = 100000000
        z_coord_max = -100000000
        z_coord_min = 100000000
        u_max = -100000000
        u_min = 100000000
        v_max = -100000000
        v_min = 100000000
        w_max = -100000000
        w_min = 100000000
        T_max = -100000000
        T_min = 100000000
        P_max = -100000000
        P_min = 100000000

        counter = 0
        for i in filenames_all:
            if i == 'run_fluent_new_noudf_v018.jou':
                continue
            print('Currently processing: ', counter, ': ', i)
            counter += 1
            # if os.path.isfile(folderpath + 'block_data/' + 'blocked_data_' + i + '.pickle'):
            #     continue
            blocked_dict, point_dict = self.initialize_dictionaries()

            start_time = time.time()

            count += 1

            if count == 0:
                if not os.path.exists(folderpath):
                    os.makedirs(folderpath)

                f = open(filepath, "w")
                f.write("File Opened\n")
            else:
                f = open(filepath, "a")

            if self.testing:
                count = self.files_per_proc-1

            data_folder = self.path + '/' + i + '/'

            file = glob.glob(data_folder + '*140.dat')[0]
            df = pd.read_csv(file)
            df.columns = df.columns.str.replace(' ', '')

            coordinates = np.concatenate((df['x-coordinate'].to_numpy().reshape(-1, 1), df['y-coordinate'].to_numpy(
            ).reshape(-1, 1), df['z-coordinate'].to_numpy().reshape(-1, 1)), axis=1)

            if self.num_var > 1:
                solution = np.concatenate((df[self.var_names[0]].to_numpy().reshape(-1, 1), df[self.var_names[1]].to_numpy(
                ).reshape(-1, 1)), axis=1)
            else:
                solution = df[self.var_names[0]].to_numpy().reshape(-1, 1)

            for nm in range(2, self.num_var):
                solution = np.concatenate((solution, df[self.var_names[nm]].to_numpy().reshape(-1, 1)), axis=1)

            power_map_file = glob.glob(data_folder + "power_map.npy")[0]
            power_map = np.load(power_map_file)

            x_num = 200
            xxx, yyy = np.meshgrid(np.linspace(0.0, 0.016, x_num), np.linspace(0.0, 0.016, x_num))
            orig_pts = np.concatenate((xxx.flatten().reshape(-1, 1), yyy.flatten().reshape(-1, 1)), 1)
            nx = 64
            grid_x, grid_y = np.meshgrid(np.linspace(0.0, 0.016, nx), np.linspace(0.0, 0.016, nx))
            new_points = np.concatenate((grid_x.flatten().reshape(-1, 1), grid_y.flatten().reshape(-1, 1)), 1)
            power_map_interp = griddata(orig_pts, power_map, new_points).flatten()
            # power_map_interp_img = power_map_interp.reshape((nx, nx))
            # fig = plt.figure()
            # ax = fig.add_subplot(121)
            # ax.imshow(power_map_interp_img)
            # ax = fig.add_subplot(122)
            # ax.imshow(power_map.reshape((x_num, x_num)))

            self.source.append(power_map_interp)
            if "boundary_condition" in self.sim_features:
                bcs_blocked = self.ml_proc.bcs_blocking(self.bcs[count, :])

            if "geometry" in self.sim_features:
                lv_blocked = self.ml_proc.geometry_blocked(lv, coordinates, df["cell-zone"].to_numpy().reshape(-1, 1))
                subdomain_type = self.ml_proc.get_subdomain_type(lv_blocked)
                whichSubdomains = subdomain_type["fluid"]
            else:
                whichSubdomains = []

            if "sourceterm_global" in self.sim_features or "sourceterm_local" in self.sim_features:
                if "sourceterm_global" in self.sim_features:
                    # pwr_blocked = self.ml_proc.source_blocking_global(self.source[count, :].reshape(1, -1),
                    #                                                whichSubdomains, self.tot_subdomains)
                    pwr_blocked = self.ml_proc.source_blocking_global(self.source[count].reshape(1, -1),
                                                                      whichSubdomains, self.tot_subdomains)
                elif "sourceterm_local" in self.sim_features:
                    a = self.ml_proc.source_blocking_local()
                    raise NotImplementedError
            if "geometry_global" in self.sim_features:
                idx = ((coordinates[:, 0] > 0) & (coordinates[:, 0] < 16000 * 10**-6)) & \
                      ((coordinates[:, 1] > 0) & (coordinates[:, 1] < 16000 * 10 ** -6)) & \
                      ((coordinates[:, 2] > 0) & (coordinates[:, 2] < 16000 * 10 ** -6))
                coordinates_hs = coordinates[idx, :]
                # lv_blocked_old = self.ml_proc.geometry_blocked(lv, coordinates_hs, df["cell-zone"].to_numpy().reshape(-1, 1)[idx])
                lv_blocked = self.ml_proc.geometry_blocked_global(lv, coordinates_hs, df["cell-zone"].to_numpy().reshape(-1, 1)[idx])

                # subdomain_type = self.ml_proc.get_subdomain_type(lv_blocked)
                # whichSubdomains = subdomain_type["solid"]
############################## TODO: GEOMETRY GLOBAL AS FEATURE
            if self.sampling:
                if not self.sampling_indices_list:
                    self.sampling_indices_list = self.get_sampling_indices(coordinates, binner)
            else:
                self.sampling_indices_list = np.arange(solution.shape[0])

            interpolated_solution = self.interpolate_solution(interp, coordinates[self.sampling_indices_list, :],
            solution[self.sampling_indices_list, :], self.domain_size)

            if self.plot_interp:
                mt = MatplotlibPlotter()
                data = {}

                for j in range(self.num_var):
                    interp1 = interpolated_solution.reshape(self.num_var, self.domain_size[0], self.domain_size[1],
                                                            self.domain_size[2])
                    data["Truth"] = interp1[j, :]
                    data["Predicted"] = interp1[j, :]
                    # folder_path = "output/process_data/"+self.var_names[j]+"/"
                    folder_path = cwd_path + f"/Fluent_outputs/{data_save_folder_name}/"+self.var_names[j]+"/"
                    mt.contours_2d_plane(data, folder_path, "solution_" + str(i) + ".png", int(self.domain_size[2]/2),
                                         "xy", False)

            soln_blocked = self.ml_proc.solution_blocked(interpolated_solution)

            if "mesh_interpolation" in self.sim_features:
                raise NotImplementedError
                data_blocked = [soln_blocked, lv_blocked, pwr_blocked, bcs_blocked]

                # Interpolation is done only at solid and solid-fluid interface subdomains
                whichSubdomains = subdomain_type["solid"] + subdomain_type["interface"]
                pointData = self.ml_proc.get_points_per_subdomain(coordinates, solution, data_blocked, self.domain_size,
                                                self.subdomain_size, whichSubdomains)

            for nm in range(self.num_var):
                blocked_dict['solution'][self.var_names[nm]].append(soln_blocked[nm, :, :])

            if "geometry" in self.sim_features or "geometry_global" in self.sim_features:
                blocked_dict['geometry'].append(lv_blocked)

            if "sourceterm_global" in self.sim_features or "sourceterm_local" in self.sim_features:
                blocked_dict['sourceterm'].append(pwr_blocked)

############################## TODO: ADD GEOMETRY GLOBAL DATA
            if "boundary_condition" in self.sim_features:
                blocked_dict['boundary_condition'].append(bcs_blocked)

            if "mesh_interpolation" in self.sim_features:
                for m in range(len(pointData)):
                    for nm in range(self.num_var):
                        point_dict['solution'][self.var_names[nm]].append(pointData[m][0][nm, :])

                    if "geometry" in self.sim_features:
                        point_dict['geometry'].append(pointData[m][1])
                    if "sourceterm_global" in self.sim_features or "sourceterm_local" in self.sim_features:
                        point_dict['sourceterm'].append(pointData[m][2])
                    if "boundary_condition" in self.sim_features:
                        point_dict['boundary_condition'].append(pointData[m][3])
                    point_dict['points']['coordinates'].append(pointData[m][4][:3, :, :])
                    for nm in range(self.num_var):
                        point_dict['points'][self.var_names[nm]].append(pointData[m][4][nm+3, :, :])

            end_time = time.time()

            f.write("Time for iteration on processor %d and iteration %d is: %f" % (self.rank, count, end_time -
                                                                                    start_time))
            f.write("\n")

            f.close()
            if self.testing:
                file_tag = self.files_per_proc
            else:
                file_tag = self.rank*self.files_per_proc + count

            if "mesh_interpolation" in self.sim_features:
                # write.pickle_write(point_dict, "point_data_" + str(i[:-4]) + ".pickle", self.filepaths["point_data"])
                write.pickle_write(point_dict, "point_data_" + str(i) + ".pickle", self.filepaths["point_data"])
            ########### write.pickle_write(blocked_dict, "blocked_data_" + str(i[:-4]) + ".pickle", self.filepaths["block_data"])
            

            x_coord_max = np.amax([coordinates[:, 0].max(), x_coord_max])
            x_coord_min = np.amin([coordinates[:, 0].min(), x_coord_min])
            y_coord_max = np.amax([coordinates[:, 1].max(), y_coord_max])
            y_coord_min = np.amin([coordinates[:, 1].min(), y_coord_min])
            z_coord_max = np.amax([coordinates[:, 2].max(), z_coord_max])
            z_coord_min = np.amin([coordinates[:, 2].min(), z_coord_min])
            u_max = np.amax([blocked_dict['solution']["x-velocity"][0].max(), u_max])
            u_min = np.amin([blocked_dict['solution']["x-velocity"][0].min(), u_min])
            v_max = np.amax([blocked_dict['solution']["y-velocity"][0].max(), v_max])
            v_min = np.amin([blocked_dict['solution']["y-velocity"][0].min(), v_min])
            w_max = np.amax([blocked_dict['solution']["z-velocity"][0].max(), w_max])
            w_min = np.amin([blocked_dict['solution']["z-velocity"][0].min(), w_min])
            T_max = np.amax([blocked_dict['solution']["temperature"][0].max(), T_max])
            T_min = np.amin([blocked_dict['solution']["temperature"][0].min(), T_min])
            P_max = np.amax([blocked_dict['solution']["pressure"][0].max(), P_max])
            P_min = np.amin([blocked_dict['solution']["pressure"][0].min(), P_min])

            f = open("96_case_min_max_recording.txt", "w+")
            f.write(f'x max is: {x_coord_max},\r\n' 
                  f'x min is: {x_coord_min},\r\n'
                  f'y max is: {y_coord_max},\r\n'
                  f'y min is: {y_coord_min},\r\n'
                  f'z max is: {z_coord_max},\r\n'
                  f'z min is: {z_coord_min},\r\n'
                  f'u_max is: {u_max},\r\n'
                  f'u_min is: {u_min},\r\n'
                  f'v_max is: {v_max},\r\n'
                  f'v_min is: {v_min},\r\n'
                  f'w_max is: {w_max},\r\n'
                  f'w_min is: {w_min},\r\n'
                  f'T_max is: {T_max},\r\n'
                  f'T_min is: {T_min},\r\n'
                  f'P_max is: {P_max},\r\n'
                  f'P_min is: {P_min},\r\n'
                    )
            f.close()
            print(f'x max is: {x_coord_max},' 
                  f'x min is: {x_coord_min},'
                  f'y max is: {y_coord_max},'
                  f'y min is: {y_coord_min},'
                  f'z max is: {z_coord_max},'
                  f'z min is: {z_coord_min},'
                  f'u_max is: {u_max},'
                  f'u_min is: {u_min},'
                  f'v_max is: {v_max},'
                  f'v_min is: {v_min},'
                  f'w_max is: {w_max},'
                  f'w_min is: {w_min},'
                  f'T_max is: {T_max},'
                  f'T_min is: {T_min},'
                  f'P_max is: {P_max},'
                  f'P_min is: {P_min},')



            write.pickle_write(blocked_dict, "blocked_data_" + str(i) + ".pickle", self.filepaths["block_data"])



    def interpolate_solution(self, interp, coordinates, solution, bins):
        interpolated_solution = []
        for j in range(solution.shape[1]):
            interpolated_solution.append(interp.grid_interpolation(coordinates, solution[:, j:j+1], bins))

        return np.asarray(interpolated_solution)

    def get_sampling_indices(self, coordinates, binner):

        bins = self.domain_size
        bin_stat = binner.threeD_binning(bins, coordinates)
        ret = bin_stat[0].reshape(bins[0] * bins[1] * bins[2])
        inp = bin_stat[3]
        list = []

        for j in range(ret.shape[0]):
            if ret[j] > 0:
                ind = np.where(inp == j)
                if ret[j] < 1000:

                    for k in ind[0]:
                        list.append(k)
                else:
                    np.random.shuffle(ind[0])
                    count = 0
                    for k in ind[0]:
                        if count < 200:
                            list.append(k)
                        else:
                            break
                        count += 1

            if j % 1000 == 0.0:
                print(j)

        return list

cur_path = os.getcwd()
data_path = cur_path + '/Fluent_outputs/Fluent_outputs_v04_96/'



def generate_boundary_conditions(ncases, filepath, var_names):

    bcs_dict = {}
    for j in var_names:
        bcs_dict[j] = {}
        bcs_dict[j]["type"] = []
        bcs_dict[j]["value"] = []

    for j in var_names:
        bcs = np.loadtxt(filepath+"_"+j+".txt", delimiter=',')
        bcs_dict[j]["type"] = bcs[:, :6]
        bcs_dict[j]["value"] = bcs[:, 6:]

    bcs_all_sample = []
    for k in range(ncases):
        bcs = []
        for j in range(len(var_names)):
            bcs.append(bcs_dict[var_names[j]]["type"][k])
        for j in range(len(var_names)):
            bcs.append(bcs_dict[var_names[j]]["value"][k])

        bcs = np.asarray(bcs)
        bcs_all_sample.append(bcs.flatten())

    return np.asarray(bcs_all_sample)


case_setup = {}
case_setup["list_var_names"] = ["x-velocity", "y-velocity", "z-velocity", "pressure", "temperature"]
case_setup["num_of_vars"] = len(case_setup["list_var_names"])
# case_setup["domain_size"] = [160, 80, 80]  # size of interpolated domain in x, y, z
# case_setup["subdomain_size"] = [16, 8, 8]
case_setup["domain_size"] = [128, 128, 128] # size of interpolated domain in x, y, z
# case_setup["subdomain_size"] = [16, 16, 16]

case_setup["subdomain_size"] = [32, 32, 32]

case_setup["sampling_indices"] = []
# case_setup["filepaths"] = {cur_path + '/Fluent_outputs/ML_Solver_data/'}
case_setup["filepaths"] = {}
case_setup["filepaths"]["block_data"] = cur_path + f'/Fluent_outputs/{data_save_folder_name}/block_data/'
case_setup["filepaths"]["point_data"] = cur_path + f'/Fluent_outputs/{data_save_folder_name}/point_data/'
case_setup["preprocessor_sampling"] = False
case_setup["plot_interpolated"] = True
# case_setup["Simulation_features"] = ["geometry", "boundary_condition"]
case_setup["Simulation_features"] = ["sourceterm_global", "geometry_global", "boundary_condition"]

case_setup["num_of_solution"] = 96
if "boundary_condition" in case_setup["Simulation_features"]:
    case_setup["boundary_condition"] = generate_boundary_conditions(case_setup["num_of_solution"],
                                        "data/boundary_conditions", case_setup["list_var_names"])
Processor = FluentDataPreprocessor(data_path,0, 0, case_setup, False)
Processor.read_fluent_data()
print('done')








