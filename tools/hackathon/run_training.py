import sys
from tools.train import main as mmdet_train

config_file_path = '/home/dalya/PycharmProjects/mmdet/mmdetection/configs/hackathon/faster_rcnn_r50_fpn_1x_car_damage_1cat_1image.py'

# sys.path.append(os.getcwd())
sys.argv.append(config_file_path)

mmdet_train()
