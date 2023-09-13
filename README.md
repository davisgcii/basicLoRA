# basicLoRA

Implementing [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) from scratch as a learning exercise.

## Overview

I'm getting back into the swing of staying on top of research and I wanted to try to implement some of what I was reading this time. Parameter efficient fine tuning (PEFT) is something that has been on my mind a lot over the last month or two and I decided to start with the initial implementation -- low-rank adaptation, or LoRA.

This is a pretty simple implementation of LoRA -- I create and train a (very small) convnet in pytorch to classify images from the CIFAR10 dataset, create a `LoRALinear` class which can quickly replace linear layers in a `nn.module`, and then continue training the LoRA-adapted model on a subset of the dataset to validate that the LoRA class is working.

## Try it yourself

To try this yourself:

- clone the repo
- spin up a virtual environment with venv -> `python3 -m venv venv`
- activate your virtual environment -> `source ./venv/bin/activate`
- install dependencies -> `pip install -r requirements.txt`
- start running the cells in the notebook!
