import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from src import u2net_full


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict(img, mask):
    weights_path = './save_weights/model_best.pth'
    threshold = 0.5

    assert os.path.exists(weights_path), f"image file {weights_path} dose not exists."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.223, 0.223, 0.223), std=(0.171, 0.171, 0.171))
    ])

    origin_img = Image.formarray(img)

    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    model = u2net_full()
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

        pred_mask = np.where(pred > threshold, 1, 0)
        origin_img = np.array(origin_img, dtype=np.uint8)
        # seg_img = origin_img * pred_mask[..., None]
        # plt.imshow(seg_img)
        # plt.show()
        # cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return pred_mask*255, t_end - t_start
