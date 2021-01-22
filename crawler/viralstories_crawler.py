import urllib.robotparser
import time
import scrapy

class ViralStoriesSpider(scrapy.Spider):
    """ViralStories crawler created to extract news titles
    """
    name = 'ViralStoriesSpider'
    starting_url_ = "http://viralstories.in/"
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
        self.sections_ += response.xpath('//li[contains(@class, "menu-item")]//a/@href').extract()

        for url in self.sections_:
            if url not in self.visited_ and self.rp.can_fetch("*", url):
                yield response.follow(url = url, callback = self.parse_news)

    def parse_news(self, response):
        """Funcion que se encarga de la extracción de las paginas de noticias dentro de una seccion.

        Args:
            response ([requests]): Página web a explorar.

        Yields:
            response.follow(): Continuacion del parse para el titulo y texto de cada noticia.
        """
        self.visited_.add(response.url)
        news = response.xpath('//h2[contains(@class, "title")]//a/@href').extract()
        for url in news:
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
        self.dict_text_[title] = response.xpath('//div[contains(@class, "post-single-content box mark-links")]//p//text()').extract()
        self.dict_[title] = response.url
