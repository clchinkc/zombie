
from datetime import datetime, timedelta

import statsmodels.api as sm
from AlgoAPI import AlgoAPI_Backtest, AlgoAPIUtil


class AlgoEvent:
    def __init__(self):
        self.lasttradetime = datetime(2018,1,1)
        self.orderPairCnt = 0 
        self.osOrder = {}
        self.arrSize = 5
        self.myTakeProfit = 5
        self.arr_closeY = []
        self.arr_closeX = []
        self.instrument = "BTCUSD", "ETHUSD"
        self.myinstrument_Y = "BTCUSD"
        self.myinstrument_X = "ETHUSD"

    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()

    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync:
            # Check for open position condition
            if bd[self.myinstrument_Y]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
                self.lasttradetime = bd[self.myinstrument_Y]['timestamp']
                # Collect observations
                self.arr_closeY.append(bd[self.myinstrument_Y]['lastPrice'])
                self.arr_closeX.append(bd[self.myinstrument_X]['lastPrice'])
                # Manage observation array size
                if len(self.arr_closeY) > self.arrSize:
                    self.arr_closeY = self.arr_closeY[-self.arrSize:]
                if len(self.arr_closeX) > self.arrSize:
                    self.arr_closeX = self.arr_closeX[-self.arrSize:]
                # Fit linear regression
                if len(self.arr_closeY) >= 2 and len(self.arr_closeX) >= 2:
                    Y = self.arr_closeY
                    X = sm.add_constant(self.arr_closeX)
                    X = sm.add_constant(X)
                    model = sm.OLS(Y, X)
                    results = model.fit()
                    #self.evt.consoleLog(results.summary())
                    coeff_b, mse = results.params[1], results.mse_resid
                    # Compute current residual
                    diff = self.arr_closeY[-1] - coeff_b * self.arr_closeX[-1] - results.params[0]

                    # Trade logic
                    if diff > 0.1 * mse:
                        self.orderPairCnt += 1
                        self.openOrder(-1, self.myinstrument_Y, self.orderPairCnt, 1) # short Y
                        if coeff_b > 0:
                            self.openOrder(1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2))) # long X
                        else:
                            self.openOrder(-1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2))) # short X
                    elif diff < -0.1 * mse:
                        self.orderPairCnt += 1
                        self.openOrder(1, self.myinstrument_Y, self.orderPairCnt, 1)  #long Y
                        if coeff_b>0:
                            self.openOrder(-1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2))) # short X
                        else:
                            self.openOrder(1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2))) # long X
                            
                # check condition for close position
                myPair = self.matchPairTradeID()
                if len(myPair)>0:
                    for tradeID, tradeID2 in myPair.items():
                        # detail for tradeID
                        instrument1 = self.osOrder[tradeID]['instrument']
                        buysell1 = self.osOrder[tradeID]['buysell']
                        openprice1 = self.osOrder[tradeID]['openprice']
                        Volume1 = self.osOrder[tradeID]['Volume']
                        # detail for tradeID2
                        instrument2 = self.osOrder[tradeID2]['instrument']
                        buysell2 = self.osOrder[tradeID2]['buysell']
                        openprice2 = self.osOrder[tradeID2]['openprice']
                        Volume2 = self.osOrder[tradeID2]['Volume']
                        # compute total PL for this pair
                        pairPL = Volume1*buysell1*(bd[instrument1]['lastPrice'] - openprice1) + Volume2*buysell2*(bd[instrument2]['lastPrice'] - openprice2) 
                        # close the pair orders
                        if pairPL > self.myTakeProfit:
                            self.closeOrder(tradeID)
                            self.closeOrder(tradeID2)
                            
    def matchPairTradeID(self):
        myPair = {}
        for tradeID in self.osOrder:
            orderRef = self.osOrder[tradeID]['orderRef']
            for tradeID2 in self.osOrder:
                orderRef2 = self.osOrder[tradeID2]['orderRef']
                if orderRef==orderRef2 and tradeID!=tradeID2 and tradeID not in myPair:
                    myPair[tradeID] = tradeID2
                    break
        return myPair

    def closeOrder(self, tradeID):
        order = AlgoAPIUtil.OrderObject(
            tradeID = tradeID,
            openclose = 'close'
        )
        self.evt.sendOrder(order)

    def openOrder(self, buysell, instrument, orderRef, volume):
        order = AlgoAPIUtil.OrderObject(
            instrument=instrument,
            orderRef=orderRef,
            volume=volume,
            openclose='open',
            buysell=buysell,
            ordertype=0       #0=market_order, 1=limit_order
        )
        self.evt.sendOrder(order)

    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        self.osOrder = oo