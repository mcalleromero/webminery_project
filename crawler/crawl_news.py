import re
from datetime import date

from scrapy.crawler import CrawlerProcess
from guardian_crawler import GuardianSpider
from upworthy_crawler import UpWorthySpider
from viralstories_crawler import ViralStoriesSpider


def crawl():

    crawler = CrawlerProcess()
    crawler.crawl(GuardianSpider)
    crawler.crawl(UpWorthySpider)
    crawler.crawl(ViralStoriesSpider)
    crawler.start()

    today = date.today().strftime("%d-%m-%Y")
    
    f = open(f"../data/guardian_news_{today}.txt", "w", encoding='utf-8')
    for title, val in GuardianSpider.dict_text_.items():
        text = ' '.join(val).replace('\n', '')
        if text != '':
            f.write(f'{title}:\t{text}\n')
    f.close()

    f = open(f"../data/upworthy_news_{today}.txt", "w", encoding='utf-8')
    for title, val in UpWorthySpider.dict_text_.items():
        text = ' '.join(val).replace('\n        ', '').replace('\n', '')
        text = re.sub(r' +', ' ', text)
        if text != '':
            f.write(f'{title}:\t{text}\n')
    f.close()

    f = open(f"../data/viralstories_news_{today}.txt", "w", encoding='utf-8')
    for title, val in ViralStoriesSpider.dict_text_.items():
        text = ' '.join(val).replace('\n', '')
        text = re.sub(r' +', ' ', text)
        if text != '':
            f.write(f'{title}:\t{text}\n')
    f.close()


if __name__ == "__main__":
    crawl()
