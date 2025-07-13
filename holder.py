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
from nautilus_trader.model.orders import Order

class HodlerConfig(StrategyConfig):
    instruments: dict[str, Instrument]
    bar_types: list[BarType]

class Hodler(Strategy):
    """
    A simple buy-and-hold strategy. Buys a fixed dollar amount of each subscribed instrument at the start.
    """
    def __init__(self, config: HodlerConfig, target_weights: dict = None):
        super().__init__(config)
        self.instruments: dict[str, Instrument] = {}
        self.equity_history = [] 
        self.order_history = []  # Track all filled orders
        self.target_weights = target_weights
        self.has_invested = False
        self.price_cache = {}
        
        # Validate that target weights sum to <= 1.0 (remainder stays in USDT)
        if target_weights and sum(target_weights.values()) > 1:
            raise ValueError("Target weights must sum to <= 1.0 (remainder stays in USDT)")
        
    def on_start(self):
        for instrument in self.config.instruments.values():
            self.instruments[instrument.id] = self.cache.instrument(instrument.id)

        for bar_type in self.config.bar_types:
            self.request_bars(bar_type)
            self.subscribe_bars(bar_type)

        self.log.info(f"{self.id}: Subscribed to bars.")

    def on_bar(self, bar: Bar):
        # Update price cache for the current instrument
        instrument_id = bar.bar_type.instrument_id
        
        if instrument_id not in self.price_cache:
            self.price_cache[instrument_id] = bar.close
            return
        
        self.price_cache[instrument_id] = bar.close
        
        # Calculate total equity
        equity = self._calculate_total_equity()
        self.equity_history.append((bar.ts_event, equity))

        # Initial investment - wait until we have prices for all instruments
        if not self.has_invested and self._has_all_prices():
            self._initial_investment()
            self.has_invested = True

    def _has_all_prices(self) -> bool:
        """Check if we have price data for all instruments."""
        for instrument in self.config.instruments.values():
            if instrument.id not in self.price_cache:
                return False
        return True

    def _calculate_total_equity(self) -> float:
        """Calculate total portfolio value including cash and all positions, minus commissions."""
        account = self.portfolio.account(BINANCE_VENUE)
        total_equity = account.balance_free(USDT)
        # Subtract total commissions paid
        commissions = account.commissions() 
        # commissions: {Currency(code='USDT', precision=8, iso4217=0, name='Tether', currency_type=CRYPTO): Money(999.00096818, USDT)}

        total_commission_cost = float(commissions[USDT].as_decimal()) if USDT in commissions else 0.0
        total_equity -= total_commission_cost
        
        # Add value of all positions
        for instrument in self.config.instruments.values():
            position = self.portfolio.net_position(instrument.id)
            current_qty = float(position)
            if current_qty > 0 and instrument.id in self.price_cache:
                current_value = current_qty * float(self.price_cache[instrument.id])
                total_equity += current_value
        
        return float(total_equity)

    def _initial_investment(self):
        """Perform initial investment across all instruments based on target weights.
        Remainder stays in USDT (cash) if target weights sum to < 1.0."""
        account = self.portfolio.account(BINANCE_VENUE)
        total_equity = account.balance_free(USDT)
        
        # Subtract any commissions already paid
        commissions = account.commissions()
        total_commission_cost = float(commissions[USDT].as_decimal()) if USDT in commissions else 0.0
        total_equity -= total_commission_cost
        
        total_weight_allocated = sum(self.target_weights.values())
        usdt_weight = 1.0 - total_weight_allocated
        
        print(f"Starting initial investment with total equity: {total_equity}")
        print(f"Total commissions paid so far: ${total_commission_cost:.2f}")
        print(f"Total weight allocated to instruments: {total_weight_allocated:.2%}")
        print(f"Remaining in USDT: {usdt_weight:.2%} (${float(total_equity) * usdt_weight:.2f})")
        
        for instrument in self.config.instruments.values():
            if instrument.id in self.target_weights:
                target_weight = self.target_weights[instrument.id]
                allocation = float(total_equity) * target_weight
                current_price = float(self.price_cache[instrument.id])
                
                # instrument: CurrencyPair(id=ETHUSDT.BINANCE, raw_symbol=ETHUSDT, asset_class=CRYPTOCURRENCY, instrument_class=SPOT, quote_currency=USDT, is_inverse=False, price_precision=2, price_increment=0.01, size_precision=6, size_increment=0.000001, multiplier=1, lot_size=None, margin_init=0, margin_maint=0, maker_fee=0.001, taker_fee=0.001, info=None)
                instrument = self.instruments[instrument.id]
                commission_rate = float(instrument.taker_fee)
                quantity = round(allocation / (current_price * (1 + commission_rate)), 6)
                
                print(f"Investing in {instrument.id}:")
                print(f"  Target weight: {target_weight:.2%}")
                print(f"  Allocation: ${allocation:.2f}")
                print(f"  Price: ${current_price:.2f}")
                print(f"  Quantity (after commission): {quantity}")
                print(f"  Expected total cost: ${quantity * current_price * (1 + commission_rate):.2f}")
                
                if quantity > 0:
                    self.buy(quantity, instrument.id)

    def on_stop(self):
        self.log.info(f"{self.id}: Backtest finished.")

    def on_order_filled(self, order: Order):
        """Record filled order information for later analysis and charting."""
        # Get current price from cache or order fill price
        print(f"order: {order}")
        current_price = None
        if hasattr(order, 'avg_px_open') and order.avg_px_open:
            current_price = float(order.avg_px_open)
        elif order.instrument_id in self.price_cache:
            current_price = float(self.price_cache[order.instrument_id])
        # OrderFilled(instrument_id=BTCUSDT.BINANCE, client_order_id=O-20241101-000000-ENGINE-000-2, venue_order_id=BINANCE-2-001, account_id=BINANCE-001, trade_id=BINANCE-2-002, position_id=BTCUSDT.BINANCE-Hodler-000, order_side=BUY, order_type=MARKET, last_qty=4.270833, last_px=70_173.73 USDT, commission=299.70028182 USDT, liquidity_side=TAKER, ts_event=1730419200000000000)

        # Record order information
        order_info = {
            'timestamp': order.ts_event,  # Order fill timestamp
            'instrument_id': order.instrument_id.symbol.value,
            'side': order.order_side.name,  # BUY or SELL
            'quantity': float(order.last_qty),
            'price': float(order.last_px),
            'value': float(order.last_qty) * float(order.last_px) - float(order.commission.as_decimal()),
            'commission': float(order.commission),
            'order_id': str(order.client_order_id)
        }
        print(f"order_info: {order_info}")

        self.order_history.append(order_info)
        
        print(f"Order filled: {order_info['side']} {order_info['quantity']} {order_info['instrument_id']} at ${order_info['price']:.2f}")
        print(f"Total value: ${order_info['value']:.2f}")
        print(f"instrument: {self.instruments[order.instrument_id]}")
        self.log.info(f"{self.id}: Order filled: {order}")

    def buy(self, quantity: Decimal, instrument_id: InstrumentId) -> None:
        """
        Users simple buy method (example).
        """
        order: MarketOrder = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instruments[instrument_id].make_qty(quantity)
        )

        self.submit_order(order)