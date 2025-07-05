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
import pandas as pd
from nautilus_trader.model.orders import Order

class TimebasedIndexRebalanceConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    rebalance_period_days: int

class TimebasedIndexRebalance(Strategy):
    """
    A general index rebalance strategy for any number of assets.
    Buys assets at the start according to target weights, then periodically rebalances.
    """
    def __init__(self, config: TimebasedIndexRebalanceConfig, target_weights: dict = None):
        super().__init__(config)
        self.instrument: InstrumentId = None
        self.rebalance_period_days = self.config.rebalance_period_days
        self.last_rebalance_ts = None
        self.equity_history = []
        self.has_invested = False
        self.price_cache = {}  # Cache for current prices of all instruments
        
        # If no weights provided, use 100% for the single asset
        if target_weights is None:
            self.target_weights = {self.config.instrument_id: 1.0}
        else:
            self.target_weights = target_weights
            
        # Validate that target weights smaller then 1.0
        if sum(self.target_weights.values()) > 1:
            raise ValueError("Target weights must smaller than 1.0")

    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.request_bars(self.config.bar_type)
        self.subscribe_bars(self.config.bar_type)
        self.log.info(f"{self.id}: Subscribed to bars.")

    def on_bar(self, bar: Bar):
        # Update price cache for the current instrument
        self.price_cache[self.config.instrument_id] = bar.close
        
        # Track equity - calculate total portfolio value
        equity = self._calculate_total_equity()
        self.equity_history.append((bar.ts_event, equity))
        
        # Initial investment
        if not self.has_invested:
            account = self.portfolio.account(BINANCE_VENUE)
            total_equity = account.balance_free(USDT)
            for instrument_id, weight in self.target_weights.items():
                instrument = self.cache.instrument(instrument_id)
                allocation = total_equity * weight
                qty = round(float(allocation) / float(bar.open), 6)
                order = self.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=instrument.make_qty(qty)
                )
                self.submit_order(order)
            self.has_invested = True
            self.last_rebalance_ts = bar.ts_event
            return

        # Rebalance check
        if self.last_rebalance_ts is not None:
            dt_last = pd.to_datetime(self.last_rebalance_ts, unit="ns")
            dt_now = pd.to_datetime(bar.ts_event, unit="ns")
            if (dt_now - dt_last).days < self.rebalance_period_days:
                return

        # Rebalance logic
        self._rebalance(bar)
        self.last_rebalance_ts = bar.ts_event
        
    def on_order_filled(self, order: Order):
        print(f"Order filled: {order}")

    def _calculate_total_equity(self) -> float:
        """
        Calculate total portfolio value including cash and all positions.
        """
        account = self.portfolio.account(BINANCE_VENUE)
        total_equity = account.balance_free(USDT)
        
        # Add value of all positions
        for instrument_id in self.target_weights.keys():
            position = self.portfolio.net_position(instrument_id)
            current_qty = float(position)
            if current_qty > 0 and instrument_id in self.price_cache:
                current_value = current_qty * float(self.price_cache[instrument_id])
                total_equity += current_value
        
        return float(total_equity)

    def _rebalance(self, bar: Bar):
        """
        Rebalance the portfolio to target weights.
        """
        total_equity = self._calculate_total_equity()
        
        for instrument_id, target_weight in self.target_weights.items():
            instrument = self.cache.instrument(instrument_id)
            target_value = total_equity * target_weight
            
            # Get current position
            position = self.portfolio.net_position(instrument_id)
            current_qty = float(position)
            current_price = float(self.price_cache.get(instrument_id, bar.close))
            current_value = current_qty * current_price
            
            # Calculate the difference
            diff_value = target_value - current_value
            
            if abs(diff_value) < 1e-6:
                continue  # No rebalance needed
            
            qty = round(abs(diff_value) / current_price, 6)
            if qty < 1e-6:
                continue
            
            side = OrderSide.BUY if diff_value > 0 else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=side,
                quantity=instrument.make_qty(qty)
            )
            self.submit_order(order)
            
            print(f"Time-based rebalancing {instrument_id}: {side} {qty} units (target: ${target_value:.2f}, current: ${current_value:.2f})")

    def on_stop(self):
        self.log.info(f"{self.id}: Backtest finished.")
