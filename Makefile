crawl:
	python3 ./crawler/crawl_news.py

generate_dataset:
	python3 ./clickbait_detector/generate_dataset.py

application:
	python3 ./clickbait_detector/__main__.py