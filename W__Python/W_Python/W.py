from ._W_basic import W_basic
from ._W_dict import W_dict
from ._W_format import W_format
from ._W_io import W_io
from ._W_tools import W_tools

class W(W_basic, W_dict, W_format, W_io, W_tools):
    def __init__(self) -> None:
        super().__init__()