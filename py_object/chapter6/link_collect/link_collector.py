from urllib.request import urlopen
import re
import sys

# 正则匹配字符串链接
LINK_REGNX = re.compile("<a [^>]*href=['\"]([^'\"]+)['\"][^>]*>")


class LinkCollector:
    def __init__(self, url):
        # 根本不需要进行切割？
        # "" + urlparse(url).netloc
        self.url = url
        self.collectlink = set()
        self.visited_links = set()

    def collect_links(self, path="/"):
        full_url = self.url + path
        self.visited_links.add(full_url)

        # 打开并读取网页内容
        page = str(urlopen(full_url).read())
        # 正则表达式匹配字符串后再逐一进行标准化
        links = {self.normalize_url(path, link) for link in LINK_REGNX.findall(page)}

        # 集合的操作
        self.collectlink = links.union(self.collectlink)
        unvisited_links = links.difference(self.visited_links)

        print(unvisited_links)

    def normalize_url(self, path, link):
        if link.startswith("http://"):
            return link
        elif link.startswith("/"):
            return self.url + link
        else:
            # 再分配？可测试
            return self.url + path.rpartition('/')[0] + '/' + link


if __name__ == '__main__':
    # 收集第一个参数作为访问网页对象
    LinkCollector(sys.argv[1]).collect_links()
