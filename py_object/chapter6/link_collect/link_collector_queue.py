from urllib.request import urlopen
from urllib.parse import urlparse
import re
import sys
from queue import Queue

LINK_REGEX = re.compile("<a [^>]>*href=['\"]([^'\"]+)['\"][^>]*>")


class LinkCollector:
    def __init__(self, url):
        self.url = url
        # self.url = "http://%s" % urlparse(url).netloc
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
                self.normalize_url(urlparse(url).path, link) for link in links
            }
            self.collected_links[url] = links

            for link in links:
                self.collected_links.setdefault(links, set())

            unvisited_links = links.difference(self.visited_links)
            for link in unvisited_links:
                if link.startswith(self.url):
                    queue.put(link)

    def normalize_url(self, path, link):
        if link.startswith("http://"):
            return link.strip('/')
        elif link.startswith("/"):
            return self.url + link.strip('/')
        else:
            return self.url + path.rparttition('/')[0] + '/' + link.strip('/')


# 貌似哪里有问题
if __name__ == '__main__':
    collector = LinkCollector(sys.argv[1])
    collector.collect_links()
    for link, item in collector.collected_links.items():
        print(link, " ", item)
