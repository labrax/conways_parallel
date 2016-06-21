#!/usr/bin/env python
# -*- coding: utf-8 -*-

def gen_test(size_0, size_1, amount_threads, seed, iterations, output_file):
    _file = open(output_file, 'w')
    _file.write("{} {} {} {} {}\n".format(size_0, size_1, amount_threads, seed, iterations))
    _file.close()

def gen_iterations_range():
    _index = 1000
    for i in range(9):
        _iterations = (i+1)*1000
        _threads = 4
        gen_test(1000, 1000, _threads, 0, _iterations, 'arq_' + str(_index) + '.in')
        _index += 1

def gen_threads_range():
    _index = 2000
    _iterations = 5000
    for i in range(4):
        _threads = 2*i + 2
        gen_test(1000, 1000, _threads, 0, _iterations, 'arq_' + str(_index) + '.in')
        _index += 1

def gen_size_range():
    _index = 3000
    _iterations = 5000
    _threads = 8
    for i in range(9):
        _size = (i+1)*1000
        gen_test(_size, _size, _threads, 0, _iterations, 'arq_' + str(_index) + '.in')
        _index += 1

if __name__ == '__main__':
    gen_threads_range()
    gen_size_range()
