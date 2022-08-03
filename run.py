import os
import sys
import glob
import time

step = 0
start_time = time.time()

# check are videos exist or not
if len(glob.glob('./2D_pose_estimation/videos/pull_to_sit/*')) == 0:
    print('no video exist')
    sys.exit(0)

# 2D pose estimation
os.chdir('./2D_pose_estimation')
step += 1; print('[' + str(step) + ']', 'start 2D pose estimation')
if os.system('python -u inference.py 2>&1 >/dev/null') != 0:
    print("'python -u inference.py 2>&1 >/dev/null' failed")
    sys.exit(0)
os.chdir('../')

# copy files
step += 1; print('[' + str(step) + ']', 'copy', len(glob.glob('./2D_pose_estimation/out/jsons/*')), "json files into './2Dto3D/input'")
if os.system('cp ./2D_pose_estimation/out/jsons/* ./2Dto3D/input') != 0:
    print("'cp ./2D_pose_estimation/out/jsons/* ./2Dto3D/input' failed")
    sys.exit(0)

# delete folder ./2D_pose_estimation/frames, command it if you don't want to delete this folder
step += 1; print('[' + str(step) + ']', "delete folder './2D_pose_estimation/frames'")
if os.system('rm -r ./2D_pose_estimation/frames') != 0:
    print("'rm -r ./2D_pose_estimation/frames' failed")
    sys.exit(0)

# delete folder ./2D_pose_estimation/out, command it if you don't want to delete this folder
step += 1; print('[' + str(step) + ']', "delete folder './2D_pose_estimation/out'")
if os.system('rm -r ./2D_pose_estimation/out') != 0:
    print("'rm -r ./2D_pose_estimation/out' failed")
    sys.exit(0)
    
# 2D to 3D
os.chdir('./2Dto3D')
step += 1; print('[' + str(step) + ']', 'start 2D to 3D')
if os.system('python -u 2Dto3D.py') != 0:
    print("'python -u 2Dto3D.py' failed")
    sys.exit(0)
os.chdir('../')

# copy files
step += 1; print('[' + str(step) + ']', 'copy', len(glob.glob('./2Dto3D/output2d/*')), "npy files into './classification/samples'")
if os.system('cp ./2Dto3D/output2d/* ./classification/samples') != 0:
    print("'cp ./2Dto3D/output2d/* ./classification/samples' failed")
    sys.exit(0)

# delete files in folder ./2Dto3D/input, command it if you don't want to delete the files in this folder
step += 1; print('[' + str(step) + ']', "delete files in folder './2Dto3D/input'")
if os.system('rm ./2Dto3D/input/*') != 0:
    print("'rm ./2Dto3D/input/*' failed")
    sys.exit(0)

# delete files in folder ./2Dto3D/output2d, command it if you don't want to delete the files in this folder
step += 1; print('[' + str(step) + ']', "delete files in folder './2Dto3D/output2d'")
if os.system('rm ./2Dto3D/output2d/*') != 0:
    print("'rm ./2Dto3D/output2d/*' failed")
    sys.exit(0)

# delete files in folder ./2Dto3D/output3d, command it if you don't want to delete the files in this folder
step += 1; print('[' + str(step) + ']', "delete files in folder './2Dto3D/output3d'")
if os.system('rm ./2Dto3D/output3d/*') != 0:
    print("'rm ./2Dto3D/output3d/*' failed")
    sys.exit(0)

# pulled-to-sit classification
os.chdir('./classification')
step += 1; print('[' + str(step) + ']', 'start pulled-to-sit classification')
print()
print('*'*50)
if os.system('python -u inference.py') != 0:
    print("'python -u inference.py' failed")
    sys.exit(0)
os.chdir('../')
print('*'*50)
print()

# delete files in folder ./classification/samples, command it if you don't want to delete the files in this folder
step += 1; print('[' + str(step) + ']', "delete files in folder './classification/samples'")
if os.system('rm ./classification/samples/*') != 0:
    print("'rm ./classification/samples/*' failed")
    sys.exit(0)
    
# delete videos in folder ./2D_pose_estimation/videos/pull_to_sit/, command it if you don't want to delete the files in this folder
step += 1; print('[' + str(step) + ']', "delete files in folder './2D_pose_estimation/videos/pull_to_sit/'")
if os.system('rm ./2D_pose_estimation/videos/pull_to_sit/*') != 0:
    print("'rm ./2D_pose_estimation/videos/pull_to_sit/*' failed")
    sys.exit(0)
    
print('done, total time:', time.time() - start_time, 'sec')