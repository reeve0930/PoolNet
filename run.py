import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from networks.joint_poolnet import build_model

if __name__ == "__main__":
    net = build_model("resnet")
    net.load_state_dict(torch.load("data/final.pth"))
    net.cuda()
    net.eval()

    img_list = sorted(glob("test/*.jpg"))

    for img in img_list:
        im = cv2.imread(img)
        in_ = np.array(im, dtype=np.float32)
        im_size = tuple(in_.shape[:2])
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = in_.transpose((2, 0, 1))

        image = torch.Tensor(in_)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            image = Variable(image)
            image = image.cuda()
            preds = net(image, mode=1)
            pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
            multi_fuse = 255 * pred

            mask = np.uint8(multi_fuse)
            heat_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            heatmap_img = np.uint8(heat_mask * 0.4) + np.uint8(im * 0.6)

            cv2.imwrite(
                os.path.join(
                    "results", "{}.png".format(img.split("/")[-1].strip(".jpg"))
                ),
                heatmap_img,
            )
