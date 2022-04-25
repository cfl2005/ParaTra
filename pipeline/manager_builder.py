# !/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import abc
import yaml
import imp
import inspect
from abc import ABCMeta, abstractmethod
from src.parallel_pipeline.structure import DiGraph
from src.parallel_pipeline.pipeline import AbstractPipe, DependencyPipeline
from src.parallel_pipeline.Manager import DepPipeLineManager


def import_file(filename):

    path = os.path.abspath(os.path.dirname(filename))
    name = os.path.splitext(os.path.basename(filename))[0]

    results = imp.find_module(name, [path])
    module = imp.load_module(name, results[0], results[1], results[2])

    return module


class TaskImporter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.tasks = {}

    def import_tasks(self):
        raise NotImplementedError("implement import_tasks function!")


class ModuleTaskImporter(TaskImporter):

    def __init__(self):
        super(ModuleTaskImporter, self).__init__()

    def import_tasks(self, modulefile):
        module = import_file(modulefile)

        sub_class_list = AbstractPipe.__subclasses__()

        for sub_class in sub_class_list:

            class_name = sub_class.__name__

            has_subclass = hasattr(module, class_name)

            if has_subclass:
                task_class = getattr(module, class_name)

                self.tasks[class_name] = task_class
            else:
                continue

        return self



class ManagerBuilder(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.module_importer = ModuleTaskImporter()

    def build_controller(self, pipe_spec, tasks):
        pass

    def build_pipeline(self, spec):
        pipeline_name = spec['name']
        tasks_spec = spec['tasks']



        pipe_dic = {}
        pipe_context = {"dependence": {},
                        "results": {}}
        for task_spec in tasks_spec:
            pipe_obj, pipe_dependence= self.process_task(task_spec)
            pipe_dic[pipe_obj.pipe_name] = pipe_obj

            pipe_context["dependence"][pipe_obj.pipe_name] = pipe_dependence
            pipe_context["results"][pipe_obj.pipe_name] = None


        graph = DiGraph(len(pipe_dic))
        for pipe in pipe_dic.values():
            graph.addEdge(pipe.pipe_name, None)
        # 根据依赖关系构建有向图
        for dest_pipename in pipe_context["dependence"]:
            if pipe_context["dependence"][dest_pipename] is None:
                continue
            for src_pipename in pipe_context["dependence"][dest_pipename]:
                graph.addEdge(src_pipename, dest_pipename)

        sorted_pipes = graph.loop_toposort()
        print("经过拓扑排序后的 pipes 顺序为: {}".format(sorted_pipes))
        pipes = [pipe_dic[pipe_name] for pipe_name in sorted_pipes]

        dep_pipeline = DependencyPipeline()

        for pipe in pipes:
            dep_pipeline.add_pipe(pipe)

        dep_pipeline.init(pipe_context)

        return pipeline_name, dep_pipeline


    def process_task(self, spec):

        # print("Task parser recieve spec : {}".format(spec))
        task_name      = spec['name']
        task_classname = spec['class']

        task_params    = {}
        for param in spec['params']:
            task_params.update(param)

        task_params.update({"pipe_name":task_name})
        task_depdencies = spec['dependence']
        pipe_obj =  self.importer.tasks[task_classname](**task_params)
        pipe_dependencies = task_depdencies

        return pipe_obj, pipe_dependencies

    def build_manager(self, spec):

        modules = spec["modules"]
        self.importer = ModuleTaskImporter()
        for module in modules:
            self.importer.import_tasks(module)

        pipeline_manager = DepPipeLineManager()

        for pipeline_spec in spec['pipelines']:
            pipeline_name, dep_pipeline = self.build_pipeline(pipeline_spec)
            pipeline_manager.regist(pipeline_name, dep_pipeline)
            
        return pipeline_manager


class PythonManagerBuilder(ManagerBuilder):

    def build_manager(self, filename):
        spec_module = import_file(filename)

        return super(PythonManagerBuilder, self).build_manager(spec_module)

class YamlManagerBuilder(ManagerBuilder):

    def build_manager(self, filename):

        with open(filename, 'r') as fin:
            spec = yaml.load(fin,Loader=yaml.FullLoader)


        return super(YamlManagerBuilder, self).build_manager(spec)



def main():
    """
    :return:
    """

    yaml_file = "conf/dependcy_pipeline.yaml"
    manager_builder = YamlManagerBuilder()
    dep_pipeline_manager = manager_builder.build_manager(yaml_file)

    input = {"data":1}
    dep_pipeline_manager.process('TestPipeline', input)


if __name__ == "__main__":
    main()
