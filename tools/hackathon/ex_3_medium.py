from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch
import matplotlib.pyplot as plt
import numpy as np


## Defining the hook
class Project3channels:  # To save the feature map of a layer
    def __init__(self, projection_method='random'):
        self.projection_method = projection_method
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        if self.projection_method == 'first_channels':
            data_projected = module_out[0, :3]
            data_projected = torch.moveaxis(data_projected, 0, 2)
        elif self.projection_method == 'random':
            indices = torch.randperm(module_out.shape[1])[:3]
            data_projected = module_out[0, indices]
            data_projected = torch.moveaxis(data_projected, 0, 2)
        elif self.projection_method == 'pca':
            data_reshaped = torch.moveaxis(module_out.squeeze(), 0, 2)
            data_reshaped = torch.reshape(data_reshaped, (data_reshaped.shape[0] * data_reshaped.shape[1], data_reshaped.shape[2]))
            # cov_mat = torch.matmul(data_reshaped, torch.transpose(data_reshaped, 0, 1))
            USV = torch.pca_lowrank(data_reshaped, 3)
            data_reshaped_projected = torch.matmul(data_reshaped, USV[2])
            data_projected = torch.reshape(data_reshaped_projected, (module_out.shape[2],  module_out.shape[3], 3))
        elif self.projection_method == 'pca-centered':
            data_reshaped = torch.moveaxis(module_out.squeeze(), 0, 2)
            data_reshaped = torch.reshape(data_reshaped, (data_reshaped.shape[0] * data_reshaped.shape[1], data_reshaped.shape[2]))
            data_reshaped = data_reshaped - data_reshaped.mean(dim=0).unsqueeze(dim=0)
            # cov_mat = torch.matmul(data_reshaped, torch.transpose(data_reshaped, 0, 1))
            USV = torch.pca_lowrank(data_reshaped, 3)
            data_reshaped_projected = torch.matmul(data_reshaped, USV[2])
            data_projected = torch.reshape(data_reshaped_projected, (module_out.shape[2],  module_out.shape[3], 3))
        self.outputs.append(data_projected)

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



#################################################################

config_file = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '../../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# Load the model - with or without a pretrained net
model = init_detector(config_file, checkpoint_file, device='cuda:0')

project_channels = Project3channels(projection_method='pca-centered')

layers_list = [model.backbone.conv1, model.backbone.layer1[0].conv1, model.backbone.layer2[0].conv1, model.backbone.layer3[0].conv1, model.backbone.layer4[0].conv1]
for layer in layers_list:
    layer.register_forward_hook(project_channels)

# test a single image
img = 'demo.jpg'
img_shape = (620, 427)
result = inference_detector(model, img)

for i, layer in enumerate(layers_list):
    layer_projected = module_output_to_numpy(project_channels.outputs[i])
    plt.figure(i)
    plt.imshow(layer_projected)
    plt.set_cmap("gist_rainbow")
    plt.show()
    aaa=1
