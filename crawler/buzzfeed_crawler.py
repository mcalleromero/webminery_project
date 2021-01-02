import urllib.robotparser
import time
import scrapy

class BuzzFeedSpider(scrapy.Spider):
    name = 'BuzzFeedSpider'
    starting_url_ = "https://www.buzzfeed.com/"
    n_jobs = 8
    sections_ = []
    dict_ = {}
    dict_text_ = {}
    visited_ = set()

    def start_requests(self):
        yield scrapy.Request(url = self.starting_url_, callback = self.parse_sections)

    def parse_sections(self, response):
        """Funcion que se encarga de la extraccion de las paginas de secciones y subsecciones del diario.

        Args:
            response ([requests]): Pagina web a explorar.

        Yields:
            response.follow(): Continuacion del parse para las noticias de cada seccion.
        """
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.set_url(f"{self.starting_url_}/robots.txt")
        self.rp.read()

        self.sections_.append(self.starting_url_)
        self.sections_ += response.xpath('//a[contains(@class, "js-card__link")]/@href').extract()

        for url in self.sections_:
            if url not in self.visited_ and self.rp.can_fetch("*", url):
                yield response.follow(url = url, callback = self.parse_content)

    def parse_content(self, response):
        """Funcion que se encarga de parsear el titulo y texto de la cada noticia.

        Args:
            response ([requests]): Página web de la noticia.
        """
        # time.sleep(1)
        self.visited_.add(response.url)
        title = response.xpath('//h1[contains(@class,"title")]/text()').extract_first().strip("\n")
        self.dict_text_[title] = response.xpath('//div[contains(@itemprop, "articleBody")]//p//text()').extract()
        self.dict_[title] = response.url
