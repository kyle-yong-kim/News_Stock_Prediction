# News_Stock_Prediction
## Data crawler scripts
1. **stock_data_crawler**
    * returns stock data obtained from IEX Cloud API
2. **news_data_crawler**
    * returns news data with labels(0: stock decrease, 1: stock increase) from NewsAPI.org

## JSON data
There are two JSON files saved within the github repo.
1. **stockQuery.json**
    * key: companyName
    * value: list[dict{changePercent, date, label}]
2. **newsData_sample.json** (wrapped into dataloader for ML training, validation, testing)
    * list[dict{companyName, date, changePercent, news_title, news_description}]
### Reason for local save
As we wanted to minize API token consumption, the obtained data are stored locally, in this case, within the github repository.
Instead of our ML model calling the API for every bootup instance, it can instead statically load json through json.load(local_json_file.json).
### How to read colorized JSON (for manual data validation)
1. Open visual studio code
      ([download link])(https://code.visualstudio.com/download)
2. Open any.json file
3. Press ctrl + shift + P
4. Type "Change Language Mode"
5. Search and select "JSON"
6. Again, press ctrl + shift + P
7. Search and select "Format Document"
8. JSON is now indented propperly with colorization
