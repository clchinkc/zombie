from datetime import datetime, timedelta

import numpy as np
from AlgoAPI import AlgoAPI_Backtest, AlgoAPIUtil
from talib import MACD


class AlgoEvent:
    def __init__(self):
        self.timer = datetime(2018, 1, 1)
        self.instrument = "BTCUSD"
        self.position = 0
        self.last_tradeID = ""
        self.macd_fastperiod = 12
        self.macd_slowperiod = 26
        self.macd_signalperiod = 9

    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()

    def on_marketdatafeed(self, md, ab):
        if md.timestamp >= self.timer + timedelta(hours=24):
            # Get historical closing prices
            res = self.evt.getHistoricalBar({"instrument": self.instrument}, max(self.macd_slowperiod, self.macd_fastperiod) + self.macd_signalperiod, "D")
            arr = [res[t]['c'] for t in res]

            # Calculate MACD
            macd, macdsignal, macdhist = MACD(np.array(arr), fastperiod=self.macd_fastperiod, slowperiod=self.macd_slowperiod, signalperiod=self.macd_signalperiod)

            # Strategy logic
            if self.position == 0:
                if macd[-1] > macdsignal[-1]:  # Buy signal
                    self.open_order(1)
                elif macd[-1] < macdsignal[-1]:  # Sell signal
                    self.open_order(-1)
            else:
                if self.position > 0 and macd[-1] < macdsignal[-1]:  # Close buy position
                    self.close_order()
                elif self.position < 0 and macd[-1] > macdsignal[-1]:  # Close sell position
                    self.close_order()

            # Update timer
            self.timer = md.timestamp

    def open_order(self, buysell):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.openclose = 'open'
        order.buysell = buysell    #1=buy, -1=sell
        order.ordertype = 0  #0=market, 1=limit
        order.volume = 0.01
        self.evt.sendOrder(order)
        
    def close_order(self):
        order = AlgoAPIUtil.OrderObject()
        order.openclose = 'close'
        order.tradeID = self.last_tradeID
        self.evt.sendOrder(order)
        

    def on_bulkdatafeed(self, isSync, bd, ab):
        pass

    def on_newsdatafeed(self, nd):
        pass

    def on_weatherdatafeed(self, wd):
        pass
    
    def on_econsdatafeed(self, ed):
        pass
        
        
    def on_orderfeed(self, of):
        # when system confirm an order, update last_tradeID and position
        if of.status=="success":
            self.position += of.fill_volume*of.buysell
            if self.position==0:
                self.last_tradeID = ""
            else:
                self.last_tradeID = of.tradeID
            
    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass
