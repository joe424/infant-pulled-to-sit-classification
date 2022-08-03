import os, sys
from glob import glob

if not os.path.isdir('./out'):
    os.system('mkdir out')
if not os.path.isdir('./frames'):
    os.system('mkdir frames')

os.chdir("./lighttrack")
os.system('python -u video_to_frames.py')

if len(glob("../frames/*/*/*.jpg")) == 0:
    print("")
    print("There is no video to process...")
    sys.exit(0)

os.system("python -u demo_video_mobile_hrnet.py")

if len(glob("../out/extracted_skeletons/*/*.pkl")) == 0:
    print("")
    print("There is no skeleton in videos...")
    sys.exit(0)

#os.chdir("../GCN")
#os.system("python main.py")

#os.chdir("../")
#os.system("rm -r frames/*")
#os.system("rm -r out/*")
print("done.")
