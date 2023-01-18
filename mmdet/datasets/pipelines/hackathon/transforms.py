import cv2
import torch
import numpy as np


from ...builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


###### HOG
@PIPELINES.register_module()
class HOG:
    """apply HOG
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    '''
    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    hog = cv2.HOGDescriptor()
    im = cv2.imread(sample)
    h = hog.compute(im)
    '''
    def __init__(self):
        self.cell_size = (8, 8)  # h x w in pixels
        self.block_size = (1, 1)  # h x w in cells
        self.nbins = 9 # number of orientation bins

    def __call__(self, results):
        """Call function to make hog images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: HOG of the inpput results
        """
        cell_size = self.cell_size
        block_size = self.block_size
        nbins = self.nbins
        img = results.get('img')
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
        hog_feats = hog.compute(img) \
            .reshape(n_cells[1] - block_size[1] + 1,
                     n_cells[0] - block_size[0] + 1,
                     block_size[0], block_size[1], nbins) \
            .transpose((1, 0, 2, 3, 4))
        hog_feats_squeeze = np.squeeze(hog_feats)
        # hog_feats_squeeze_norm = np.sqrt(np.sum(hog_feats_squeeze**2,axis=2))
        hog_feats_squeeze_resize = cv2.resize(hog_feats_squeeze, (img.shape[1], img.shape[0]))
        results['img'] = hog_feats_squeeze_resize

        return results
