# !/usr/bin/python
# -*- coding: utf-8 -*-



class DepPipeLineManager(object):

    def __init__(self):
        self.pipeline_dict = {}

    def regist(self, pipeline_name, pipeline):
        self.pipeline_dict[pipeline_name] = pipeline

    def unregist(self, pipeline_name):
        if pipeline_name in self.pipeline_dict:
            del self.pipeline_dict[pipeline_name]

    def process(self, pipeline_name, input):
        self.pipeline_dict[pipeline_name].process(input)
