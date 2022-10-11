import cv2
import util.io
from torchvision.transforms import Compose
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import numpy as np
from openvino.runtime import Core
import torch

net_w = net_h = 384
transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

image = util.io.read_image('input/ball.jpg')

# img = transform({"image": image})["image"]
# img_input = np.expand_dims(img, axis=0)
# print(img_input.shape)

img_input = cv2.resize(image, (384,576)) # resize to a target input size
img_input = np.expand_dims(img_input, 0)  # add batch dimension
img_input = img_input.transpose(0, 3, 1, 2)  # convert to NCHW layout

core = Core()
net = core.read_model('midas.xml')
compiled_model = core.compile_model(net, 'CPU')

results = compiled_model.infer_new_request({0: img_input})
out = next(iter(results.values()))
print(out.shape)

out = (
    torch.nn.functional.interpolate(
        torch.tensor(out).unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    .squeeze()
    .cpu()
    .numpy()
)

print(out.shape)

util.io.write_depth(
    'res', out, bits=2, absolute_depth=False
)
