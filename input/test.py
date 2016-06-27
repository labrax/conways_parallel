#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os
import subprocess
from time import time
import difflib


test_files = os.listdir('.')
test_files.sort()
exec_files = os.listdir("..")
exec_files.sort()

speedup = True

if "serial" not in exec_files:
    print("serial executable has not been found - ignoring speedup")
    speedup = False
else:
    exec_files.remove("serial")
    # adiciona no começo da lista de execução
    exec_files.insert(0, "serial")

configured_execs = ["openmp", "openmp_tasks", "pthreads", "serial", "cuda", "cuda_shared"]


resultados_de_todos = dict()

def print_begin(prog_name, arq_teste):
    print("")
    print("-" * 5 + prog_name[3:] + " " + arq_teste + "-" * 5)


def print_end(prog_name, arq_teste, time):
    print("Time elapsed: " + str(time) + "s")
    if speedup:
        if prog_name != "../serial":
            print("Speedup: " + str(resultados_de_todos['../serial'][arq_teste]/time))
    print("")


def make():
    p = subprocess.Popen(['/usr/bin/make', '-C', '..'])
    p.wait()


def exec_program(prog_name, test_input, test_output):
    _input = open(test_input, 'r')
    _output = open(test_output, 'w')
    print_begin(prog_name, test_input)
    start = time()
    p = subprocess.Popen([prog_name], stdin=_input, stdout=_output)
    p.wait()
    end = time()
    _input.close()
    _output.close()
    _data = resultados_de_todos.get(prog_name, dict())
    _data[test_input] = end-start
    resultados_de_todos[prog_name] = _data
    print_end(prog_name, test_input, end-start)


def test_all():
    global exec_files
    prog_run = 0
    for ef in exec_files:
        if ef not in configured_execs:
            continue
        prog_run += 1
        for tf in test_files:
            if 'in' in tf:
                exec_program("../" + ef, tf, tf[:-3] + '_' + ef + '.out')
    if prog_run == 0:
        make()
        exec_files = os.listdir("..")
        test_all()


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
                last = ""
                pre_last = ""
                count = 0
                print("###", 'serial', ef, tf, "###")
                for line in difflib.unified_diff(serial_data, other_data, fromfile='serial', tofile=tf[:-3], lineterm='', n=0):
                    for prefix in ('---', '+++', '@@'):
                        if line.startswith(prefix):
                            break
                    else:
                        pre_last = last
                        last = line
                        count += 1
                        #print(line)
                count = count / 2
                if count > 0:
                    print("{} lines differ:\n{}\n{}".format(count, pre_last, last))
                else:
                    print("files are equal")
                print("######")


if __name__ == "__main__":
    if '-r' in sys.argv:
        test_all()
    if '-c' in sys.argv:
        try:
            check_diff()
        except:
            test_all()
            check_diff()
    if len(sys.argv) == 1 or ('-r' not in sys.argv and '-c' not in sys.argv):
        print("Options are -r to run tests and -c to compare")
        
