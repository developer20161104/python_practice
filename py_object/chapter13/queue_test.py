import os
from multiprocessing import Queue, Process, cpu_count


# error 有编码的问题，版本不一致
def search(paths, query_q, result_q):
    lines = []

    for path in paths:
        lines.extend(l.strip() for l in open(path))
    query = query_q.get()
    while query:
        # brute force
        result_q.put([l for l in lines if query in l])
        query = query_q.get()


if __name__ == '__main__':
    # 整体思想：分配进程来对每个文件进行穷举搜索，并将结果保存在队列中返回，结果队列是共享的
    cpus = cpu_count()
    pathnames = [f for f in os.listdir(os.getcwd()) if os.path.isfile(f)]
    paths = [pathnames[i::cpus] for i in range(cpus)]
    query_queues = [Queue() for p in range(cpus)]
    results_queue = Queue()

    search_proc = [
        Process(target=search, args=(p, q, results_queue))
        for p, q in zip(paths, query_queues)
    ]

    for proc in search_proc:
        proc.start()

    for q in query_queues:
        q.put('def')
        q.put(None)

    for i in range(cpus):
        for match in results_queue.get():
            print(match)

    for proc in search_proc:
        proc.join()

