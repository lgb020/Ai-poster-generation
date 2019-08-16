import torch
from torchvision import transforms
from transformer_net import TransformerNet
import re
from PIL import Image
import os

def stylize(content_image_path, pathout, model, device):
    r"""
    :param content_image_path: the path of content image
            model: path of the model, default:./saved_models/starry-night.model
    :return: saved stylize_image
    """
    # args.content_image='../images/content-images/test1.jpg'

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = Image.open(content_image_path)
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    output = model(content_image).cpu()
    img = output[0].clone().clamp(0, 255).numpy()
    # img = output[0].clone().clamp(0, 255).detach().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(pathout)

# mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'
# model_path = mainroot + 'saved_models/plyr.pth'
# device = torch.device("cuda")
# with torch.no_grad():
#     style_model = TransformerNet()
#     state_dict = torch.load(model_path)
#     # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
#     for k in list(state_dict.keys()):
#         if re.search(r'in\d+\.running_(mean|var)$', k):
#             del state_dict[k]
#     style_model.load_state_dict(state_dict)
#     style_model.to(device)

# if __name__ == '__main__':
#     content_image_path = mainroot + '1.jpg'
#     pathout = mainroot + '2.jpg'
#     stylize(content_image_path, pathout=pathout, model=style_model)
