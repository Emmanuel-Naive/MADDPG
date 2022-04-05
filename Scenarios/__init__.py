'''
Functions for reading data according to the given scenario name
'''
from imp import load_source
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return load_source('', pathname)