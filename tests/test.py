#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import subprocess
from time import time
import difflib


test_files = os.listdir('.')
exec_files = os.listdir("..")
configured_execs = ["openmp", "openmp_tasks", "pthreads", "serial", "cuda"]

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
    for ef in exec_files:
        if ef not in configured_execs:
            continue
        for tf in test_files:
            if 'in' in tf:
                exec_program("../" + ef, tf, tf[:-3] + '_' + ef + '.out')


# based on: http://stackoverflow.com/questions/19120489/compare-two-files-report-difference-in-python
def check_diff():
    for tf in test_files:
        if 'in' in tf:
            _f_serial = open(tf[:-3] + '_' + 'serial' + '.out', 'r')
            serial_data = _f_serial.read().strip().splitlines()
            _f_serial.close()
            for ef in exec_files:
                if ef not in configured_execs or ef == 'serial':
                    continue
                _f_other = open(tf[:-3] + '_' + ef + '.out', 'r')
                other_data = _f_other.read().strip().splitlines()
                _f_other.close()
                print("###", 'serial', ef, tf, "###")
                for line in difflib.unified_diff(serial_data, other_data, fromfile='serial', tofile=tf[:-3], lineterm='', n=0):
                    for prefix in ('---', '+++', '@@'):
                        if line.startswith(prefix):
                            break
                    else:
                        print(line)
                print("######")
        


    
    
if __name__ == "__main__":
    #test_all()
    check_diff()
