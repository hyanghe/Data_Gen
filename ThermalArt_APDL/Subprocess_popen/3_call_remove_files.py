import subprocess 
import os
txt_name = "/commands_tile_heating_HTC_BC_area_effects.txt"
# dest_fpath = "/home/hhe/APDL_Workspace/DecayCurveData/One_tile_3D_data_cluster_division400_random/"
dest_fpath = "./One_tile_3D_data_cluster_division400_random/"
for folder in os.listdir(os.path.dirname(dest_fpath)):
    print("Currently processing: ", folder)
    temp_working_dir = dest_fpath + folder + '/'  
    input_file = os.path.dirname(temp_working_dir) + txt_name
    print("Directory has been changed to: ", temp_working_dir)
    print("Solving...")
    p = subprocess.Popen(['python', 'remove_files.py', '-f', temp_working_dir])
