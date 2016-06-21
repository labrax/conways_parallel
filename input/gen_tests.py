#!/usr/bin/env python
# -*- coding: utf-8 -*-

def gen_test(size_0, size_1, amount_threads, seed, iterations, output_file):
    _file = open(output_file, 'w')
    _file.write("{} {} {} {} {}\n".format(size_0, size_1, amount_threads, seed, iterations))
    _file.close()

def gen_all_tests():
    _index = 1000
    for i in range(9):
        _iterations = (i+1)*1000
        _threads = 4
        gen_test(1000, 1000, _threads, 0, _iterations, 'arq_' + str(_index) + '.in')
        _index += 1

if __name__ == '__main__':
    gen_all_tests()
