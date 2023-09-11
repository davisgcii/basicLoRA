import torch
import torchvision
import torchvision.transforms as transforms  # common image transformations that can be chained together using Compose

# create a transform object that converts an image to a tensor and normalizes its pixel values
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)  # normalize pixel values of image
