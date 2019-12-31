import argparse
import os
import models
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='../../demo/results/frames')
parser.add_argument('-d1','--dir1', type=str, default='../../demo/src/frames')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
files.sort()
all_dist=0
for file in files:
	if(file[-3:]!="jpg"):
		continue
	file_num=file[5:]
	d1_file_name=file_num
	if(os.path.exists(os.path.join(opt.dir1,d1_file_name))):
		# Load images
		img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = util.im2tensor(util.load_source_image(os.path.join(opt.dir1,d1_file_name)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = model.forward(img0,img1)
		print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
		all_dist+=dist01
print("总距离：%.3f"%all_dist)
f.writelines("总距离：%.3f\n"%all_dist)
avg_dist=all_dist/len(files)
print("平均距离：%.3f"%avg_dist)
f.writelines("平均距离：%.3f"%avg_dist)
f.close()
