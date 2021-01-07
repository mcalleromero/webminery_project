import time
from datetime import date

from scrapy.crawler import CrawlerProcess
from guardian_crawler import GuardianSpider
from upworthy_crawler import UpWorthySpider


def crawl():

    crawler = CrawlerProcess()
    # crawler.crawl(GuardianSpider)
    crawler.crawl(UpWorthySpider)
    crawler.start()

    today = date.today().strftime("%d-%m-%Y")
    
    # f = open(f"./data/guardian_news_{today}.txt", "w", encoding='utf-8')
    # for title, val in GuardianSpider.dict_text_.items():
    #     f.write(f'{title}:\t{val}\n')
    # f.close()

    f = open(f"./data/upworthy_news_{today}.txt", "w", encoding='utf-8')
    for title, val in UpWorthySpider.dict_text_.items():
        f.write(f'{title}:\t{val}\n')
    f.close()


if __name__ == "__main__":
    crawl()
