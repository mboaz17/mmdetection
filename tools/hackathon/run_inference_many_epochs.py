from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import glob

if 1:  # fasterrcnn
    config_file = '../../configs/hackathon/faster_rcnn_r50_fpn_1x_car_damage_1cat_1image.py'
    checkpoint_file_path = '/home/dalya/PycharmProjects/mmdet/mmdetection/results/1cat_1image_faster/1cat_1image_frozen1/'
else:  # retinanet
    config_file = '../../configs/hackathon/retinanet_r50_fpn_1x_car_damage_1cat_1image.py'
    checkpoint_file_path = '/home/dalya/PycharmProjects/mmdet/mmdetection/results/1cat_1image_retina/1cat_1image_frozen1/'


for epoch_ind in range(1, 51, 1):
    checkpoint_file = os.path.join(checkpoint_file_path, 'epoch_{}.pth'.format(epoch_ind))
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image
    img_path = '/home/dalya/PycharmProjects/mmdet/mmdetection/data/car_damage/train/79.jpg'
    result = inference_detector(model, img_path)

    # show the results
    show_result_pyplot(model, img_path, result, score_thr=0.5, title='epoch #{}'.format(epoch_ind))
    aaa=1
