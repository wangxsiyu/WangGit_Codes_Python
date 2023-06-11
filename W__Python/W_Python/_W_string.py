import numpy as np

class W_string():
    def W_str_numbers_separated_by_underscore(lst):
        for i in range(len(lst)):
            if i == 0:
                tstr = f"{lst[0]}"
            else:
                tstr = f"{tstr}_{lst[i]}"
        return tstr