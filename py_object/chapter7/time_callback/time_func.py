import datetime
import time


class TimedEvent:
    def __init__(self, endtime, callback):
        self.endtime = endtime
        # 将函数当成一个参数调用
        self.callback = callback

    def ready(self):
        # 判断是否到了时间
        return self.endtime <= datetime.datetime.now()


class Timer:
    def __init__(self):
        # 就绪事件列表
        self.events = []

    def call_after(self, delay, callback):
        # 为新的callback设定时间
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=delay)

        # 添加新事件
        self.events.append(TimedEvent(end_time, callback))

    def run(self):
        while True:
            # 逐一执行已经准备就绪的方法
            ready_events = (e for e in self.events if e.ready())
            for event in ready_events:
                event.callback(self)
                self.events.remove(event)
            time.sleep(0.5)