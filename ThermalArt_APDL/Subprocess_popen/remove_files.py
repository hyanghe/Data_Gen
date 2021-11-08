import argparse
import os
# print('##################################################')

import argparse
import random

# type determines the type of the argument

# parser = argparse.ArgumentParser()
   
# parser.add_argument('-n', type=int, required=True, 
#     help="define the number of random integers")
# args = parser.parse_args()

# n = args.n

# for i in range(n):
#     print(random.randint(-100, 100))



# parser = argparse.ArgumentParser(description='delete folder')
# parser.add_argument('-f', dest='folder', type=int)  # store argument in args.something
# # ... etc etc etc more options
# args = parser.parse_args()
# folder = args.f
# # This is the beef: once the arguments are parsed, pass them on
# print(folder)



parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', dest='folder', type=str, required=True, help='designate folder for file remove')
args = parser.parse_args()
temp_working_dir = args.folder
print('in remove: ', temp_working_dir)
# def remove(temp_working_dir):
os.chmod(temp_working_dir, 0o777)
#     return 'Removing ######################################'
    # Delete unuseful files
files_in_directory = os.listdir(temp_working_dir)
print('files_in_directory: ', files_in_directory)
filtered_files = [file for file in files_in_directory if not file.endswith(".txt")]
print('filtered_files: ', filtered_files)
for root, dirs, files in os.walk(temp_working_dir):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o777)
    for f in files:
        os.chmod(os.path.join(root, f), 0o777)
for file in filtered_files:
    path_to_file = os.path.join(temp_working_dir, file)
    os.remove(path_to_file)

    # print("finished processing: ", folder)

# if __name__ == '__main__':
# remove(temp_working_dir)
print('done')
