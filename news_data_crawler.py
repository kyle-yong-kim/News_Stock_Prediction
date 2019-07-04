import requests
import json
import datetime

# this function can be tried at a later time
def stringDecode(inpStr):
    inpStr = inpStr.encode("utf-8").decode("utf-8").replace("\r\n","")
    return inpStr

def queryStringBuilder(companyName, date, topN, sortMethod):
    # string to datetime conversion
    topN = 10
    date_end = datetime.datetime.strptime(date, '%Y-%m-%d')
    date_start = date_end + datetime.timedelta(days=-1)
    date_from = date_start.strftime('%Y-%m-%d')
    date_to = date_end.strftime('%Y-%m-%d')

    api_key = "16b7dc00706b4e95bbc56367429bab0e"
    news_API_url = f"https://newsapi.org/v2/everything?apiKey={api_key}&q={companyName}&from={date_from}&to={date_to}&pageSize={2}&language=en&sortBy={sortMethod}"

    return news_API_url

def queryNewsArticle(companyName, queryString, label, date, changePercent):
    r1 = requests.get(queryString)
    r1_json = json.loads(r1.text)['articles']

    newsList = [{"companyName": companyName, "date": date, "changePercent": changePercent, "title": x["title"], "description": x["description"], "content": x["content"], "label": label} for x in r1_json]
    return newsList

def loadStockJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def mainLoop():
    newsResult = []

    # take only top 10 relevant news articles to the search query
    topN = 10

    # first, load the stock json obtained from stock_data_crawler
    fileLoc = "C:\\Temp\\stockQuery.json"
    stockJson = loadStockJson(fileLoc)

    # put a break point just for testing json stuff
    i = 0
    for companyName, stockData in stockJson.items():
        i += len(stockData)
        
        # since we are broke and can't pay for pro api version rip
        if i > 220:
            break

        for item in stockData:
            queryString = queryStringBuilder(companyName, item["date"], topN, "relevancy")
            newsResult += queryNewsArticle(companyName, queryString, item["label"], item['date'], item['changePercent'])

    jsonResult = json.dumps(newsResult)

    with open("C:\\Temp\\newsData.json", "w") as f:
        f.write(jsonResult)

mainLoop()

print("news data loading completed")