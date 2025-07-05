from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy
from decimal import Decimal

class HodlerConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType

class Hodler(Strategy):
    """
    A simple buy-and-hold strategy. Buys a fixed dollar amount of each subscribed instrument at the start.
    """
    def __init__(self, config: HodlerConfig, target_weights: dict = None):
        super().__init__(config)
        self.instruments: Instrument = None
        self.equity_history = []  # Add this line
        self.has_invested = False

    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.request_bars(self.config.bar_type)
        self.subscribe_bars(self.config.bar_type)
        self.log.info(f"{self.id}: Subscribed to bars.")

    def on_bar(self, bar: Bar):
        equity = self.portfolio.account(BINANCE_VENUE).balance_free(USDT) + self.portfolio.net_exposure(self.config.instrument_id)
        self.equity_history.append((bar.ts_event, equity))
        if not self.has_invested:
            self.buy(round(self.portfolio.account(BINANCE_VENUE).balance_free(USDT)/bar.high, 2))
            # self.buy(1)
            self.has_invested = True

    def on_stop(self):
        self.log.info(f"{self.id}: Backtest finished.")

    def buy(self, quantity: Decimal) -> None:
        """
        Users simple buy method (example).
        """
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity)
        )

        self.submit_order(order)