# !/usr/bin/python
# -*- coding: utf-8 -*-


import time
import random
from src.parallel_pipeline.pipeline import AbstractPipe

class DataTransformPipe(AbstractPipe):

    def __init__(self, indicator, pipe_name):
        super(DataTransformPipe, self).__init__(pipe_name=pipe_name)
        self.indicator = indicator

    def do_process(self, input):
        result = self.indicator + input['data']
        time.sleep(random.randint(1, 3))
        print("Data transform {} entit indicator: {}".format( self.pipe_name, self.indicator + input['data']))
        input["data"] = result
        return input

    def to_dict(self):
        dic = {
            "pipe_name": self.pipe_name,
            "indicator": self.indicator
        }
        return dic

class MapPipe(AbstractPipe):
    def __init__(self, add_unit):
        super(MapPipe, self).__init__()
        self.add_unit = add_unit

    def do_process(self, input):

        input['data'] = input['data'] + self.add_unit
        print("Map pipe add unit: {}, result: {}".format(self.add_unit, input['data']))

        return input

    def to_dict(self):
        dic = {
            "pipe_name": self.pipe_name,
            "add_unit": self.add_unit
        }

        return dic


class ReducePipe(AbstractPipe):
    def __init__(self):
        super(ReducePipe, self).__init__()

    def do_process(self, input):
        print("Reduce pipe 接收到的内容为: {}".format(input))
        if not type(input) is list:
            inputs = [input]
        else:
            inputs = input

        sum = 0
        for input in inputs:
            sum += input['data']

        result = {"data": sum}

        print("Reduce Pipe result is {}".format(result))

        return result

    def to_dict(self):
        dic = {
            "pipe_name": self.pipe_name
        }

        return dic
