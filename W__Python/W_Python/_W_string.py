import numpy as np
import re

class W_string():
    def W_str_numbers_separated_by_underscore(lst):
        for i in range(len(lst)):
            if i == 0:
                tstr = f"{lst[0]}"
            else:
                tstr = f"{tstr}_{lst[i]}"
        return tstr
    
    def W_find_substr(mstr, ptn):
        return [m.start() for m in re.finditer(ptn, mstr)]

    def W_str_select_between_patterns(mstr, pt1 = None, pt2 = None, n1 = 1, n2 = 1):
        if pt1 is None:
            p1 = 0
        else:
            p1 = W_string.W_find_substr(mstr, pt1)[n1-1] + 1
        if pt2 is None:
            p2 = len(mstr)
        else:
            p2 = W_string.W_find_substr(mstr, pt2)[n2-1]
        return mstr[p1:p2]