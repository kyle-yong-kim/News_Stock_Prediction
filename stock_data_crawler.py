import requests
import json

def queryStock(stockSymb, IEX_API_token, crit_dev):
    filteredDict = {}
    for symbol in stockSymb:
        # this returns a JSON file
        r1 = requests.get(f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/1m" + IEX_API_token)

        # r2 is query for company's searchable Name, in this case A
        r2 = requests.get(f"https://cloud.iexapis.com/stable/stock/{symbol}/stats/companyName" + IEX_API_token)
        r1_json = json.loads(r1.text)
        companyName = json.loads(r2.text)

        # initialize list within filter dictionary
        filteredDict[companyName] = []

        # create list from r.text date and changePercent indices
        dateList = [(x["date"], x["changePercent"]) for x in r1_json]

        for date, change in dateList:
            if abs(change) >= crit_dev:
                filteredDict[companyName].append(
                    {"date": date, "changePercent": change, "label": 1 if (change>=crit_dev) else 0})
    
    return filteredDict

def testSymbol(stockSymb, IEX_API_token):

    # define invalid symbol list
    invalid = []

    for symbol in stockSymb:
        try:
            r2 = requests.get(f"https://cloud.iexapis.com/stable/stock/{symbol}/stats/companyName" + IEX_API_token)
            companyName = json.loads(r2.text)
        except:
            invalid.append(symbol)

    return invalid

def mainLoop():
    # define static critical stock deviation percentage values 2%
    crit_dev = 1.5
    IEX_API_token = "?token=pk_cfc8dbb58b6247b598a448d1eff42a7f"
    stock_symb_string = "XOM CVX COP SLB EOG KMI PSX OXY MPC VLO ECL APD SHW LYB NEM PPG BLL VMC IP NUE BA HON UNP UTX LMT MMM GE UPS CAT CSX AMZN DIS HD CMCSA NFLX MCD BKNG NKE SBUX CHTR WMT PG KO PEP PM COST MO MDLZ EL CL JNJ PFE UNH MRK ABT MDT TMO AMGN LLY DHR BRK.B JPM BAC WFC C AXP USB MS GS BLK MSFT AAPL GOOG FB V MA CSCO INTC ORCL GOOGL T VZ CTL NEE DUK D SO EXC AEP SRE XEL PEG ED AMT CCI PLD SPG EQIX PSA WELL AVB EQR VTR"
    # stock_symb_string = "AAPL"

    # searchable stock symbol
    stockSymb = stock_symb_string.split(" ")

    # test stock symbol's validity, and delete symbol off if data is corrupted.
    invalidList = testSymbol(stockSymb, IEX_API_token)
    for inval in invalidList:
        stockSymb.remove(inval)
    
    # query IEX Cloud for stock data
    queryResult = queryStock(stockSymb, IEX_API_token, crit_dev)
    jsonResult = json.dumps(queryResult)

    with open("C:\\Temp\\stockQuery.json", "w") as f:
        f.write(jsonResult)

mainLoop()

print("data loading completed")