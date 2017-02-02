# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.fish import fish
import numpy as np

fish_path = '/home/GRAPHICS2/20d_bar/kaggle/FISH'
for split in ['train', 'test']:
    name = '{}_{}'.format('fish', split)
    __sets[name] = (lambda split=split: fish(split, fish_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
