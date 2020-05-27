from multiprocessing import Process, cpu_count
from threading import Thread
import time
import os


# 通过实现一个子类来实现子进程（类似于Thread实现子线程）
# 但是由于全局解释锁（GIL）的存在，会使得不能同时存在两个线程访问CPU，因此此处则相当于逐一执行
# class MuchCPU(Thread):
class MuchCPU(Process):
    def run(self) -> None:
        print(os.getpid())

        for i in range(200000000):
            pass


if __name__ == '__main__':
    proc = [MuchCPU() for f in range(cpu_count())]

    t = time.time()
    for p in proc:
        p.start()

    for p in proc:
        p.join()

    print('work took {} seconds'.format(time.time()-t))