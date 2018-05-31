# -*- coding: utf-8 -*-

import os

# global namespace:
from packtml import clustering
from packtml import decision_tree
from packtml import metrics
from packtml import neural_net
from packtml import recommendation
from packtml import regression
from packtml import utils

# set the version
packtml_location = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(packtml_location, "VERSION")) as vsn:
    __version__ = vsn.read().strip()

# remove from global namespace
del os
del packtml_location
del vsn

__all__ = [
    'clustering',
    'decision_tree',
    'metrics',
    'neural_net',
    'recommendation',
    'regression',
    'utils'
]
