"""
NNumpy
======

Neural network library using numpy.

This package is part of the Deep Learning & Neural Nets lecture material.
The following copyright statement applies to all code within this package.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form,
requires explicit prior acceptance of the authors.
"""

__version__ = "2.0.0"

__author__ = "Pieter-Jan Hoedt"
__email__ = "hoedt@ml.jku.at"
__copyright__ = "Copyright 2019-2022, JKU Institute for Machine Learning"

from . import initialisers as init

from .base import *
from .containers import *
from .connections import *
from .pooling import *
from .activations import *
from .losses import *
from .optimisers import *
from .normalisation import *
from .regularisation import *
from .reductions import *
