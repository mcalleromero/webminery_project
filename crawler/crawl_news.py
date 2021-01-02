import time
from datetime import date

from guardian_crawler import GuardianSpider
from scrapy.crawler import CrawlerProcess


def crawl():

    starttime_guardian = time.time()

    crawler2 = CrawlerProcess()
    crawler2.crawl(GuardianSpider)
    crawler2.start()

    endtime_guardian = time.time() - starttime_guardian

    today = date.today().strftime("%d-%m-%Y")

    f = open(f"./data/guardian_news_{today}.txt", "w")
    for title, val in GuardianSpider.dict_text_.items():
        f.write(f'{title}:\t{val}\n')
    f.close()

    print(f'El crawler de The Guardian ha tardado: {endtime_guardian} segundos')


if __name__ == "__main__":
    crawl()
