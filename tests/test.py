#!env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import subprocess
from time import time


def exec_program(prog_name, test_input, test_output):
    _input = open(test_input, 'r')
    _output = open(test_output, 'w')
    start = time()
    p = subprocess.Popen([prog_name], stdin=_input, stdout=_output)
    p.wait()
    end = time()
    _input.close()
    _output.close()
    print(prog_name, test_input, test_output, end-start)


def test_all():
    test_files = os.listdir()
    exec_files = os.listdir("..")
    
    for ef in exec_files:
        if ef.endswith('out'):
            for tf in test_files:
                if 'in' in tf:
                    exec_program("../" + ef, tf, tf[:-3] + '_' + ef[:-4] + '.out')
    
    
if __name__ == "__main__":
    test_all()
