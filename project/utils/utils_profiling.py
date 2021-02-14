"""
Source code originally from SE(3)-Transformer (https://github.com/FabianFuchsML/se3-transformer-public/)
"""

try:
    profile
except NameError:
    def profile(func):
        return func