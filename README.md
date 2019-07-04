# News_Stock_Prediction
## Data crawler scripts]
1. stock_data_crawler
    * returns stock data obtained from IEX Cloud API
2. news_data_crawler
    * returns news data with labels(0: stock decrease, 1: stock increase) from NewsAPI.org

## JSON data
As we wanted to minize API token consumption, the obtained data are stored locally, in this case, within the github repository.
Instead of our ML model calling the API, it can instead json.load(local_json_file.json).
