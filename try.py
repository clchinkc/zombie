from datetime import datetime, timedelta

import numpy as np
from AlgoAPI import AlgoAPI_Backtest, AlgoAPIUtil
from talib import RSI


class AlgoEvent:
    def __init__(self):
        self.timer = datetime(2018, 1, 1)
        self.instrument = "BTCUSD"
        self.position = 0
        self.last_tradeID = ""
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.trade_volume_percentage = 0.01  # Percentage of equity to trade

    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()

    def on_marketdatafeed(self, md, ab):
        # Update timer and check for a new day
        if md.timestamp >= self.timer + timedelta(hours=24):
            # Use md attributes for strategy logic
            # Example: midPrice to calculate RSI
            res = self.evt.getHistoricalBar({"instrument": self.instrument}, self.rsi_period + 1, "D")
            arr = [res[t]['c'] for t in res]
            rsi = RSI(np.array(arr), self.rsi_period)

            # Trading logic
            if self.position == 0:
                if rsi[-1] > self.rsi_overbought:
                    self.open_order(-1, ab['NAV'])
                elif rsi[-1] < self.rsi_oversold:
                    self.open_order(1, ab['NAV'])
            else:
                if self.position > 0 and rsi[-1] < 50:
                    self.close_order(ab['NAV'])
                elif self.position < 0 and rsi[-1] > 50:
                    self.close_order(ab['NAV'])

            self.timer = md.timestamp

    def calculate_trade_volume(self, nav):
        return nav * self.trade_volume_percentage

    def open_order(self, buysell, nav):
        volume = self.calculate_trade_volume(nav)
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.openclose = 'open'
        order.buysell = buysell
        order.ordertype = 0
        order.volume = volume
        self.evt.sendOrder(order)

    def close_order(self, nav):
        volume = self.calculate_trade_volume(nav)
        order = AlgoAPIUtil.OrderObject()
        order.openclose = 'close'
        order.tradeID = self.last_tradeID
        order.volume = volume
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
        if of.status == "success":
            self.position += of.fill_volume * of.buysell
            if self.position == 0:
                self.last_tradeID = ""
            else:
                self.last_tradeID = of.tradeID
            
    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass