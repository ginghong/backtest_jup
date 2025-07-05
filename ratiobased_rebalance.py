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


class RatioBasedIndexRebalanceConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    rebalance_threshold: float

class RatioBasedIndexRebalance(Strategy):
    """
    A ratio-based index rebalance strategy that triggers rebalancing when 
    portfolio weights deviate from target weights by a specified threshold.
    """
    def __init__(self, config: RatioBasedIndexRebalanceConfig, target_weights: dict = None):
        super().__init__(config)
        self.instrument: InstrumentId = None
        self.rebalance_threshold = self.config.rebalance_threshold
        self.equity_history = []
        self.has_invested = False
        self.price_cache = {}  # Cache for current prices of all instruments
        
        # If no weights provided, use 100% for the single asset
        if target_weights is None:
            self.target_weights = {self.config.instrument_id: 1.0}
        else:
            self.target_weights = target_weights
            
        # Validate that target weights sum to 1
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
        
        # Track equity
        equity = self.portfolio.account(BINANCE_VENUE).balance_free(USDT) + self.portfolio.net_exposure(self.config.instrument_id)
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
            return

        # Check if rebalancing is needed
        if self._should_rebalance():
            self._rebalance(bar)

    def on_order_filled(self, order: Order):
        print(f"Order filled: {order}")

    def _should_rebalance(self) -> bool:
        """
        Check if the portfolio weights have deviated from target weights
        beyond the specified threshold.
        """
        current_weights = self._calculate_current_weights()
        
        for instrument_id, target_weight in self.target_weights.items():
            current_weight = current_weights.get(instrument_id, 0.0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > self.rebalance_threshold:
                print(f"Rebalance triggered: {instrument_id} deviation = {deviation:.4f} (threshold: {self.rebalance_threshold:.4f})")
                print(f"Current weight: {current_weight:.4f}, Target weight: {target_weight:.4f}")
                return True
        
        return False

    def _calculate_current_weights(self) -> dict:
        """
        Calculate the current weights of each asset in the portfolio.
        """
        account = self.portfolio.account(BINANCE_VENUE)
        total_equity = account.balance_free(USDT)
        
        # Calculate total portfolio value including all positions
        for instrument_id in self.target_weights.keys():
            position = self.portfolio.net_position(instrument_id)
            current_qty = float(position)
            if current_qty > 0 and instrument_id in self.price_cache:
                current_value = current_qty * float(self.price_cache[instrument_id])
                total_equity += current_value
        
        if total_equity == 0:
            return {}
        
        # Calculate current weights
        current_weights = {}
        for instrument_id in self.target_weights.keys():
            position = self.portfolio.net_position(instrument_id)
            current_qty = float(position)
            if current_qty > 0 and instrument_id in self.price_cache:
                current_value = current_qty * float(self.price_cache[instrument_id])
                current_weights[instrument_id] = current_value / float(total_equity)
            else:
                current_weights[instrument_id] = 0.0
        
        return current_weights

    def _rebalance(self, bar: Bar):
        """
        Rebalance the portfolio to target weights.
        """
        account = self.portfolio.account(BINANCE_VENUE)
        total_equity = account.balance_free(USDT)
        
        # Calculate total portfolio value
        for instrument_id in self.target_weights.keys():
            position = self.portfolio.net_position(instrument_id)
            current_qty = float(position)
            if current_qty > 0 and instrument_id in self.price_cache:
                current_value = current_qty * float(self.price_cache[instrument_id])
                total_equity += current_value
        
        # Rebalance each asset
        for instrument_id, target_weight in self.target_weights.items():
            instrument = self.cache.instrument(instrument_id)
            target_value = float(total_equity) * target_weight
            
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
            
            print(f"Rebalancing {instrument_id}: {side} {qty} units (target: ${target_value:.2f}, current: ${current_value:.2f})")

    def on_stop(self):
        self.log.info(f"{self.id}: Backtest finished.")