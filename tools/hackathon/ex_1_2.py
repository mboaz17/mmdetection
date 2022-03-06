from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch
import matplotlib.pyplot as plt
import numpy as np


## Defining the hooks
class SaveFilter:  # To save the weights of a layer
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module.weight)

    def clear(self):
        self.outputs = []

class SaveOutput:  # To save the feature map of a layer
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def module_output_to_numpy(tensor):  # convert a tensor to numpy
    return tensor.detach().to('cpu').numpy()

def visualize_data(input, n=64, rows=8, cols=8, figsize=(20, 20), resize=None):  # visualize data
    # with plt.style.context("seaborn-white"):
    plt.figure(figsize=figsize, frameon=False)
    for idx in range(n):
        plt.subplot(rows, cols, idx+1)
        if resize:
            img = mmcv.imresize(input[idx, 0], resize)
        else:
            img = input[idx, 0]
        plt.imshow(img)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()



############################# Exercise 1 ####################################

save_filter = SaveFilter()

config_file = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_files = [None, '../../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth']

for checkpoint_file in checkpoint_files:

    # Load the model - with or without a pretrained net
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    for layer in model.modules():  # registering the hook for every conv layer
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            layer.register_forward_hook(save_filter)

    # test a single image
    img = 'demo.jpg'
    result = inference_detector(model, img)

    ## Visualise some filters - number 1 and number 7
    filters = module_output_to_numpy(save_filter.outputs[0])
    visualize_data(filters)
    filters = module_output_to_numpy(save_filter.outputs[6])
    visualize_data(filters)

    save_filter.clear()  # clear the hook stack




############################# Exercise 2 ####################################

save_output = SaveOutput()
layer = model.rpn_head.rpn_cls
layer.register_forward_hook(save_output)

# test a single image
img = 'demo.jpg'
img_shape = (620, 427)
result = inference_detector(model, img)

for fpn_level in range(0, 5):  # show objectness maps for every fpn-level and every anchor
    outputs = module_output_to_numpy(save_output.outputs[fpn_level])
    outputs = np.moveaxis(outputs, 0, 1)
    visualize_data(outputs, n=3, rows=1, cols=3, figsize=(20, 5), resize=img_shape)


# show the detections on the image
show_result_pyplot(model, img, result)

