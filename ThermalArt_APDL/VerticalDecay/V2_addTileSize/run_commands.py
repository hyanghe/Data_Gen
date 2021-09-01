"/appls/ProductInstalls/ANSYS/ansys_inc/v201/ansys/bin/mapdl"  -g -p ansys -smp -np 4 -lch -dir "/nfs/sjord45.data/span/TM_support/Jimin/template_ models/new_1um_model_200430/new_tile_power_model" -j "new_tile_power_model" -s read -l en-us -t -d X11 | tee -i 'new_tile_power_model.out'

 

"C:\ANSYSDev\ANSYS Inc\v201\ANSYS\bin\winx64\MAPDL.exe", -dir "E:\APDL_Workspace\test_run_commands" -j decaycurve -i "commands_tile_heating_HTC_BC_area_effects.txt"

"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe"  -g -p ansys -dis -mpi INTELMPI -np 2 -lch -dir "E:\APDL_Workspace" -j "DecayCurve" -s read -l en-us -t -d win32   

"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe"  -g -p ansys -dis -mpi INTELMPI -np 2 -lch -dir "E:\APDL_Workspace\test_run_commands" -j "DecayCurve1" -s read -l en-us -t -d win32   


"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe" -dis -mpi INTELMPI -np 2 -dir "E:\APDL_Workspace\test_run_commands" -j "DecayCurve1" -s read -t -d win32   

"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe" -B -dis -mpi INTELMPI -np 2 -dir "E:\APDL_Workspace\test_run_commands" -j "DecayCurve1" -i "commands_tile_heating_HTC_BC_area_effects.txt" -o "output.txt"


"C:\ANSYSDev\ANSYS Inc\v201\ansys\bin\winx64\MAPDL.exe" -B -dis -np 8 -dir "E:\APDL_Workspace\DecayCurveData_20210624_near_far_dig\test_run" -j "DCurveModel" -i "commands_tile_heating_HTC_BC_area_effects_Haiyang_1.txt" -o "output.txt"
        