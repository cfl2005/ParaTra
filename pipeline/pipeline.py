# !/usr/bin/python
# -*- coding: utf-8 -*-


import time
import random
#import Queue
import threading
from abc import ABCMeta, abstractmethod
from threading import Condition, Thread
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor



class CountDownLatch:

    def __init__(self, count):
        self.count = count
        self.condition = Condition()

    def await(self):
        try:
            self.condition.acquire()
            while self.count > 0:
                self.condition.wait()
        finally:
            self.condition.release()

    def countDown(self):
        try:
            self.condition.acquire()
            self.count -= 1
            self.condition.notifyAll()
        finally:
            self.condition.release()


class AbstractPipe(object):

    def __init__(self, pipe_name=None,  pipe_context=None):
        self.pipe_name = pipe_name
        self.next_pipe = None
        self.pipe_context = pipe_context

    def set_next(self, next_pipe):
        self.next_pipe = next_pipe

    def init(self, pipe_context):
        self.pipe_context = pipe_context

    def shut_down(self, timeout, time_unit):

    def process(self, input):
        # try:
        out = self.do_process(input)

        if 'results' in self.pipe_context:
            self.pipe_context['results'][self.pipe_name] = out

        # 如果正确输出，并且当前正确定义了下一个pipe,调用下一个pipeline
        if out and self.next_pipe:
            self.next_pipe.process(out)

    def do_process(self, input):
        raise NotImplementedError("Please implement do_process in inherit pipe class!")


class Function():
    __metaclass__ = ABCMeta

    def __init__(self, params={}, result={}, nlu_template=None,  nlg_template=None ):
        self.params = {}
        self.result = {}
        self.nlu_template = nlu_template
        self.nlg_tempalte = nlg_template

    def process(self, input):
        raise NotImplementedError("Please implement Function`s process logical")


    def gen_nlu_pattern(self):
        return self.nlu_template


    def gen_nlg_pattern(self):
        return self.nlg_tempalte

    def __call__(self, input):
        self.process(input)

class FunctionPipe(AbstractPipe):
    __metaclass__ =  ABCMeta

    def __init__(self, pipe_name, function):
        super(FunctionPipe, self).__init__(pipe_name=pipe_name)
        self.function = function

    @abstractmethod
    def do_process(self, inputs):
        """
        :param inputs:
        :return:
        """
        kwargs = dict([(param_name, self.pipe_context[param_name]) \
                       for  param_name in self.function.params])

        result = self.function.execute(**kwargs)

        for res_name in self.function.res_names:
            self.pipe_context[res_name] = result[res_name]

        std_nlu = None
        nlg = None

        return std_nlu , nlg

class Constraint(Function):
    __metaclass__ =  ABCMeta

    def __init__(self,type_):
        self.type_ = type_

    def do_process(self, input):
        self.fit(input)

    @abstractmethod
    def fit(self,input):
        raise NotImplementedError("Please implement in inherit class!")



class ThreadPipeDecorator(AbstractPipe):

    def __init__(self, delegate_pipe, pool_executor):
        """
        :param delegate_pipe:
        :param pool_executor:
        """
        self.delegate_pipe = delegate_pipe
        self.thread_pool = pool_executor

    def init(self, pipe_context):
        self.delegate_pipe.init(pipe_context)

    def process(self, input):
        print("当前 pipe thread recive input: {}".format(input))

        task = lambda input: self.delegate_pipe.process(input)

        self.thread_pool.submit(task, input)

        # 使用单线程 提交任务
        # thread = threading.Thread(target=task, args=[input,])
        # thread.setDaemon(True)
        # thread.start()


    def set_next(self, next_pipe):

        self.delegate_pipe.set_next(next_pipe)


class WorkerPipeDecorator(AbstractPipe):

    def __init__(self, delegate_pipe, pool_executor):
        """ """
        super(WorkerPipeDecorator, self).__init__()
        self.delegate_pipe = delegate_pipe
        self.thread_pool = pool_executor
        self.queue = Queue.Queue()          # 内部队列
        self.__active = False
        # 启动pipe worker
        self.start()


    def process(self, input):
        """
        :param input:
        :return:
        """
        event = {"type": "pipe", "data": {
            "context": self.pipe_context,
            "input": input
        }}
        self.queue.put(event)


    def start(self,):
        """
        启动 thread pipe
        :return:
        """
        self.__active = True
        def task():
            """
            当前线程
            :return:
            """
            print("start pipe")
            while self.__active:
                event = self.queue.get(block=True, timeout=100)

                pipe_context =  event['data']["context"]
                input = event['data']['input']
                self.delegate_pipe.init(pipe_context)
                result = self.delegate_pipe.do_process(input)

                event["data"]['input'] = result

                self.next_pipe.queue.put(event)

        self.thread_pool.submit(fn=task)

    def shut_down(self):
        self.__active = False


class ParallelPipe(AbstractPipe):

    def __init__(self, pipe_name=None,  pool=None):
        super(ParallelPipe, self).__init__(pipe_name=pipe_name)
        self.pipes = []
        self.count_down = None
        self.pool = pool if pool else ThreadPoolExecutor(max_workers=3)

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def init(self, pipe_context):
        for pipe in self.pipes:
            pipe.init(pipe_context)
        self.count_down = CountDownLatch(len(self.pipes))

    def do_process(self, input):
        """
        并行执行 内部保存的各个子 pipe ， 所有pipe 执行完成才执行下游 pipe
        :param input:
        :return:
        """

        def task(pipe, input, count_down, callback=None):
            """
            将 pipe 的执行与 同步锁 count_down 包装在一起
            """
            # count_down.wait()
            result = pipe.do_process(input)
            if callback:
                callback(result)
            count_down.countDown()
            return result

        # results = []
        futures = []
        for pipe in self.pipes:
            input_cp = {"data": input['data']}
            future = self.pool.submit(task, pipe, input_cp, self.count_down)
            futures.append(future)

        self.count_down.await()
        results = [future.result() for future in futures]

        return results



class SimplePipeline(AbstractPipe):
    def __init__(self, pool_executor):
        self.thread_pool = pool_executor
        self.pipes = []

    def process(self, input):
        first_pipe = self.pipes[0]
        first_pipe.process(input)

    def init(self, pipe_context):
        prev_pipe = self
        self.pipe_context = pipe_context

        for pipe in self.pipes:
            prev_pipe.set_next(pipe)
            prev_pipe = pipe
            pipe.init(pipe_context)

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def addAsThreadPoolBasedPipe(self, pipe):
        self.add_pipe(ThreadPipeDecorator(pipe, self.thread_pool))

    def addAsWokerBasedPipe(self, pipe):
        self.add_pipe(WorkerPipeDecorator(pipe, self.thread_pool))


class DependencyPipeline(AbstractPipe):

    def __init__(self, ):
        self.pipes = []

    def init(self, pipe_context):
        self.pipe_context = pipe_context

        for pipe in self.pipes:
            pipe.init(pipe_context)

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def dependency_check(self, pipe):
        is_check = True
        dependencies = self.pipe_context['dependence'][pipe.pipe_name]

        if dependencies and len(dependencies) > 0:
            for dep_pipename in dependencies:
                if self.pipe_context["results"][dep_pipename] is None:
                    print("pipe {} result is None".format(dep_pipename))
                    is_check = is_check and False

        return is_check

    def process(self, inputs):
        for pipe in self.pipes:
            is_check = self.dependency_check(pipe)
            if not is_check:
                print("Pipe {} dependency is not statisified, Please check it!")
            pipe.process(inputs)

    def reset(self):
        for pipe_name in self.pipe_context['results']:
            self.pipe_context['results'][pipe_name] = None


# ---------------  pipe ----------------------------------------------------------

class DataTransformPipe(AbstractPipe):

    def __init__(self, indicator):
        super(DataTransformPipe, self).__init__()
        self.indicator = indicator

    def do_process(self, input):
        result = self.indicator + input['data']
        time.sleep(random.randint(1, 3))
        print("Data transform entit indicator: {}".format(self.indicator + input['data']))
        input["data"] = result
        return input


class MapPipe(AbstractPipe):
    def __init__(self, add_unit):
        super(MapPipe, self).__init__()
        self.add_unit = add_unit

    def do_process(self, input):

        input['data'] = input['data'] + self.add_unit
        print("Map pipe add unit: {}, result: {}".format(self.add_unit, input['data']))

        return input


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


def main():
    pool = ThreadPoolExecutor(max_workers=20)
    simple_pipeline = SimplePipeline(pool_executor=pool)

    pipe_one = DataTransformPipe(indicator=1)
    pipe_two = DataTransformPipe(indicator=2)
    pipe_three = DataTransformPipe(indicator=3)
    pipe_four = DataTransformPipe(indicator=4)
    pipe_five = DataTransformPipe(indicator=5)


    paral_pipe = ParallelPipe()
    for i in range(10):
        paral_pipe.add_pipe(MapPipe(i))

    reduce_pipe = ReducePipe()

    pipes = [pipe_one, pipe_two, pipe_three, pipe_four, pipe_five, paral_pipe, reduce_pipe]

    for pipe in pipes:
        simple_pipeline.addAsThreadPoolBasedPipe(pipe)
        # simple_pipeline.addAsWokerBasedPipe(pipe)

    simple_pipeline.init(pipe_context={})

    for i in range(10):
        simple_pipeline.process(input={'data': 10 * i})


    while True:
        time.sleep(2)

if __name__ == "__main__":
    main()
