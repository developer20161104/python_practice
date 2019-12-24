from urllib.request import urlopen
from urllib.parse import urlparse
import re
import sys
from queue import Queue

# 正则匹配：a 任意个除开>外的元素 href= 选择两者的一个 至少一个除开'"外的元素 选择两者中的一个 任意个除开>外的元素 结束
LINK_REGEX = re.compile("<a [^>]*href=['\"]([^'\"]+)['\"][^>]*>")


# 采用dict；来存储相关的链接并放入集合中，自动过滤重复
class LinkCollector:
    def __init__(self, url):
        self.url = "http://%s" % urlparse(url).netloc
        # 此处使用字典收集链接之间的关联
        self.collected_links = {}
        self.visited_links = set()

    def collect_links(self):
        queue = Queue()
        queue.put(self.url)

        while not queue.empty():
            url = queue.get().rstrip('/')
            # 加入访问集合
            self.visited_links.add(url)
            page = str(urlopen(url).read())
            links = LINK_REGEX.findall(page)

            links = {
                # 规范化链接
                self.normalize_url(urlparse(url).path, link) for link in links
            }
            self.collected_links[url] = links

            for link1 in links:
                self.collected_links.setdefault(link1, set())

            unvisited_links = links.difference(self.visited_links)
            for link2 in unvisited_links:
                if link2.startswith(self.url):
                    queue.put(link2)

    def normalize_url(self, path, link3):
        # 在去除尾部的斜杠后会出现端口号与下级页面位置重合，产生冲突
        if link3.startswith("http://"):
            return link3
        elif link3.startswith("/"):
            return self.url + link3
        else:
            return self.url + path.rpartition('/')[0] + '/' + link3


if __name__ == '__main__':
    # collector = LinkCollector("http://localhost:8000")
    collector = LinkCollector(sys.argv[1])
    collector.collect_links()
    for link, item in collector.collected_links.items():
        print(link, " ", item)
