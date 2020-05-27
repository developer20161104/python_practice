from threading import Thread
import json
from urllib.request import urlopen
import time


CITIES = ['Edmonton',
          'Toronto','Regina','Victoria''Winnipeg']


class TempGetter(Thread):
    def __init__(self, city):
        super().__init__()
        self.city = city


    def run(self) -> None:
        # API的使用需要进行注册，用不了了
        url_temp = (
            'http://api.openweathermap.org'
            'find?q={},CA'
        )

        response = urlopen(url_temp.format(self.city))
        data = json.loads(response.read().decode())

        self.temperature = data['main']['temp']


if __name__ == '__main__':

    threads = [TempGetter(c) for c in CITIES]
    start = time.time()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    for thread in threads:
        print("it is {0.temperature:.0f}* C in {0.city}".format(thread))

        print("Got {} temps in {} seconds".format(len(threads), time.time()-start))

