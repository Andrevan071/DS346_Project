# Data Science 346 project

## Web Scraper
The webscraper for stackoverflow questions and answers related to data science, machine learning and artificial intelligence
is implemented in the Webscraper directory. Each scraped category is contained in the respective json files. The
data-science tagged questions is contained in the Webscraper/data/data_science.json file, the machine-learning
tagged questions is contained in the Webscraper/data/machine_learning.json file and the artificial-intelligence
tagged questions is contained in the Webscraper/data/artificial_intelligence.json file.

To execute the webscraper, path into the Webscraper directory, by running `cd Webscraper`.

To run the webscraper for the data-science tagged questions, execute the following command: `python webscraper.py --tag ds`
or `python3 webscraper.py --tag ds`.

To run the webscraper for the machine-learning tagged questions, execute the following command: `python webscraper.py --tag ml`
or `python3 webscraper.py --tag ml`.

To run the webscraper for the artificial-intelligence tagged questions, execute the following command: `python webscraper.py --tag ai`
or `python3 webscraper.py --tag ai`.