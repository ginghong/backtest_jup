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
    instruments: dict[str, Instrument]
    bar_types: list[BarType]
    rebalance_threshold: float

class RatioBasedIndexRebalance(Strategy):
    """
    A ratio-based index rebalance strategy that triggers rebalancing when 
    portfolio weights deviate from target weights by a specified threshold.
    """
    def __init__(self, config: RatioBasedIndexRebalanceConfig, target_weights: dict = None):
        super().__init__(config)
        self.instruments: dict[str, Instrument] = {}
        self.rebalance_threshold = self.config.rebalance_threshold
        self.equity_history = []
        self.order_history = []  # Track all filled orders
        self.has_invested = False
        self.price_cache = {}  # Cache for current prices of all instruments
        self.target_weights = target_weights
        self.pending_orders = 0  # Track pending orders to avoid race conditions
        self.rebalance_in_progress = False
        
        # Validate that target weights sum to <= 1
        if target_weights and sum(target_weights.values()) > 1:
            raise ValueError("Target weights must be smaller than 1.0")

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
        
        # Track equity
        equity = self._calculate_total_equity()
        self.equity_history.append((bar.ts_event, equity))
        
        # Initial investment - wait until we have prices for all instruments
        if not self.has_invested and self._has_all_prices():
            self._initial_investment(bar)
            self.has_invested = True
            return

        # Don't start new rebalancing if one is in progress
        if self.rebalance_in_progress or self.pending_orders > 0:
            return

        # Check if rebalancing is needed
        if self._should_rebalance():
            self._rebalance(bar)

    def _has_all_prices(self) -> bool:
        """Check if we have price data for all instruments."""
        for instrument in self.config.instruments.values():
            if instrument.id not in self.price_cache:
                return False
        return True

    def on_order_filled(self, order: Order):
        """Record filled order information for later analysis and charting."""
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
        
        self.order_history.append(order_info)
        
        print(f"Order filled: {order_info['side']} {order_info['quantity']} {order_info['instrument_id']} at ${order_info['price']:.2f}")
        print(f"Total value: ${order_info['value']:.2f}, Commission: ${order_info['commission']:.2f}")
        
        self.pending_orders = max(0, self.pending_orders - 1)
        if self.pending_orders == 0:
            self.rebalance_in_progress = False

    def _calculate_total_equity(self) -> float:
        """
        Calculate total portfolio value including cash and all positions, minus commissions.
        """
        account = self.portfolio.account(BINANCE_VENUE)
        total_equity = account.balance_free(USDT)
        
        # Subtract total commissions paid
        commissions = account.commissions()
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

    def _get_available_balance(self) -> float:
        """Get current available USDT balance, accounting for ongoing orders."""
        account = self.portfolio.account(BINANCE_VENUE)
        return float(account.balance_free(USDT))

    def _initial_investment(self, bar: Bar):
        """Perform initial investment across all instruments based on target weights."""
        available_balance = self._get_available_balance()
        
        # Subtract any commissions already paid for more accurate calculation
        account = self.portfolio.account(BINANCE_VENUE)
        commissions = account.commissions()
        total_commission_cost = float(commissions[USDT].as_decimal()) if USDT in commissions else 0.0
        available_balance -= total_commission_cost
        
        print(f"Starting initial investment with available balance: {available_balance}")
        print(f"Total commissions paid so far: ${total_commission_cost:.2f}")
        
        # Calculate actual allocations considering commission costs
        planned_orders = []
        total_cost = 0
        
        for instrument in self.config.instruments.values():
            if instrument.id in self.target_weights:
                target_weight = self.target_weights[instrument.id]
                target_allocation = available_balance * target_weight
                current_price = float(self.price_cache[instrument.id])
                
                instrument_obj = self.instruments[instrument.id]
                commission_rate = float(instrument_obj.taker_fee)
                
                # Correct commission calculation: allocation after commission
                net_allocation = target_allocation / (1 + commission_rate)
                quantity = net_allocation / current_price
                
                # Apply instrument precision
                precision = instrument_obj.size_precision
                quantity = round(quantity, precision)
                
                if quantity > 0:
                    actual_cost = quantity * current_price * (1 + commission_rate)
                    planned_orders.append({
                        'instrument': instrument_obj,
                        'quantity': quantity,
                        'cost': actual_cost,
                        'weight': target_weight,
                        'price': current_price
                    })
                    total_cost += actual_cost

        # Check if total cost exceeds available balance
        if total_cost > available_balance:
            print(f"⚠️ Total planned cost ${total_cost:.2f} exceeds available balance ${available_balance:.2f}")
            # Scale down all orders proportionally
            scale_factor = available_balance * 0.99 / total_cost  # 1% buffer
            for order_plan in planned_orders:
                order_plan['quantity'] *= scale_factor
                order_plan['cost'] *= scale_factor
            print(f"🔧 Scaled down orders by factor {scale_factor:.4f}")

        # Execute orders sequentially with balance checks
        for order_plan in planned_orders:
            current_balance = self._get_available_balance()
            
            if order_plan['cost'] > current_balance:
                print(f"⚠️ Insufficient balance for {order_plan['instrument'].id}: need ${order_plan['cost']:.2f}, have ${current_balance:.2f}")
                continue
                
            print(f"Investing in {order_plan['instrument'].id}:")
            print(f"  Target weight: {order_plan['weight']:.2%}")
            print(f"  Allocation: ${order_plan['cost']:.2f}")
            print(f"  Price: ${order_plan['price']:.2f}")
            print(f"  Quantity: {order_plan['quantity']}")
            
            order = self.order_factory.market(
                instrument_id=order_plan['instrument'].id,
                order_side=OrderSide.BUY,
                quantity=order_plan['instrument'].make_qty(order_plan['quantity'])
            )
            self.submit_order(order)
            self.pending_orders += 1

    def _should_rebalance(self) -> bool:
        """
        Check if the portfolio weights have deviated from target weights
        beyond the specified threshold.
        """
        current_weights = self._calculate_current_weights()
        
        for instrument in self.config.instruments.values():
            if instrument.id in self.target_weights:
                target_weight = self.target_weights[instrument.id]
                current_weight = current_weights.get(instrument.id, 0.0)
                deviation = abs(current_weight - target_weight)
                
                if deviation > self.rebalance_threshold:
                    print(f"Rebalance triggered: {instrument.id} deviation = {deviation:.4f} (threshold: {self.rebalance_threshold:.4f})")
                    print(f"Current weight: {current_weight:.4f}, Target weight: {target_weight:.4f}")
                    return True
        
        return False

    def _calculate_current_weights(self) -> dict:
        """
        Calculate the current weights of each asset in the portfolio.
        """
        total_equity = self._calculate_total_equity()
        
        if total_equity == 0:
            return {}
        
        # Calculate current weights
        current_weights = {}
        for instrument in self.config.instruments.values():
            if instrument.id in self.target_weights:
                position = self.portfolio.net_position(instrument.id)
                current_qty = float(position)
                if current_qty > 0 and instrument.id in self.price_cache:
                    current_value = current_qty * float(self.price_cache[instrument.id])
                    current_weights[instrument.id] = current_value / float(total_equity)
                else:
                    current_weights[instrument.id] = 0.0
        
        return current_weights

    def _rebalance(self, bar: Bar):
        """
        Rebalance the portfolio to target weights with improved order sequencing.
        """
        if self.rebalance_in_progress:
            return
            
        self.rebalance_in_progress = True
        total_equity = self._calculate_total_equity()
        
        # Plan all rebalancing trades first
        rebalance_plans = []
        
        # Rebalance each asset
        for instrument in self.config.instruments.values():
            if instrument.id in self.target_weights:
                target_weight = self.target_weights[instrument.id]
                instrument_obj = self.instruments[instrument.id]
                target_value = float(total_equity) * target_weight
                
                # Get current position
                position = self.portfolio.net_position(instrument.id)
                current_qty = float(position)
                
                # Use proper price from cache with fallback
                if instrument.id in self.price_cache:
                    current_price = float(self.price_cache[instrument.id])
                else:
                    print(f"⚠️ No price cached for {instrument.id}, skipping rebalance")
                    continue
                    
                current_value = current_qty * current_price
                
                # Calculate the difference
                diff_value = target_value - current_value
                
                if abs(diff_value) < 1e-6:
                    continue  # No rebalance needed
                
                # Account for commission when calculating quantity
                commission_rate = float(instrument_obj.taker_fee)
                
                if diff_value > 0:  # Need to buy
                    # For buying: net purchase amount = diff_value / (1 + commission)
                    net_purchase = diff_value / (1 + commission_rate)
                    qty = net_purchase / current_price
                    side = OrderSide.BUY
                else:  # Need to sell
                    # For selling: quantity to sell directly
                    qty = abs(diff_value) / current_price
                    side = OrderSide.SELL
                
                # Apply instrument precision
                precision = instrument_obj.size_precision
                qty = round(qty, precision)
                
                if qty < 1e-8:  # Skip tiny trades
                    continue
                
                rebalance_plans.append({
                    'instrument': instrument_obj,
                    'quantity': qty,
                    'side': side,
                    'target_value': target_value,
                    'current_value': current_value,
                    'diff_value': diff_value
                })

        # Separate buy and sell orders
        sell_orders = [p for p in rebalance_plans if p['side'] == OrderSide.SELL]
        buy_orders = [p for p in rebalance_plans if p['side'] == OrderSide.BUY]
        
        # Calculate funding requirements
        total_buy_cost = sum(
            p['quantity'] * float(self.price_cache[p['instrument'].id]) * (1 + float(p['instrument'].taker_fee))
            for p in buy_orders
        )
        total_sell_proceeds = sum(
            p['quantity'] * float(self.price_cache[p['instrument'].id]) * (1 - float(p['instrument'].taker_fee))
            for p in sell_orders
        )
        available_balance = self._get_available_balance()
        
        # Check if we need additional sells to fund buy orders
        funding_shortfall = total_buy_cost - (available_balance + total_sell_proceeds)
        
        if funding_shortfall > 0:
            print(f"💰 Need additional ${funding_shortfall:.2f} funding for buy orders")
            additional_sells = self._create_funding_sells(funding_shortfall)
            sell_orders.extend(additional_sells)
        
        # Execute all sell orders first
        for plan in sell_orders:
            order = self.order_factory.market(
                instrument_id=plan['instrument'].id,
                order_side=plan['side'],
                quantity=plan['instrument'].make_qty(plan['quantity'])
            )
            self.submit_order(order)
            self.pending_orders += 1
            print(f"Rebalancing {plan['instrument'].id}: {plan['side']} {plan['quantity']} units (target: ${plan['target_value']:.2f}, current: ${plan['current_value']:.2f})")
        
        # Then execute buy orders
        for plan in buy_orders:
            order = self.order_factory.market(
                instrument_id=plan['instrument'].id,
                order_side=plan['side'],
                quantity=plan['instrument'].make_qty(plan['quantity'])
            )
            self.submit_order(order)
            self.pending_orders += 1
            print(f"Rebalancing {plan['instrument'].id}: {plan['side']} {plan['quantity']} units (target: ${plan['target_value']:.2f}, current: ${plan['current_value']:.2f})")

        if self.pending_orders == 0:
            self.rebalance_in_progress = False

    def _create_funding_sells(self, funding_needed: float) -> list:
        """
        Create additional sell orders to fund buy orders when balance is insufficient.
        """
        additional_sells = []
        total_equity = self._calculate_total_equity()
        funding_raised = 0
        
        # Find positions that are above target weight and can be sold
        overweight_candidates = []
        
        for instrument in self.config.instruments.values():
            if instrument.id in self.target_weights and instrument.id in self.price_cache:
                target_weight = self.target_weights[instrument.id]
                target_value = total_equity * target_weight
                
                position = self.portfolio.net_position(instrument.id)
                current_qty = float(position)
                current_price = float(self.price_cache[instrument.id])
                current_value = current_qty * current_price
                
                # If current value exceeds target, it's a candidate for additional selling
                excess_value = current_value - target_value
                if excess_value > 1.0:  # Only consider meaningful excess
                    overweight_candidates.append({
                        'instrument': self.instruments[instrument.id],
                        'excess_value': excess_value,
                        'current_qty': current_qty,
                        'current_price': current_price
                    })
        
        # Sort by excess value (most overweight first)
        overweight_candidates.sort(key=lambda x: x['excess_value'], reverse=True)
        
        # Create sells from overweight positions
        for candidate in overweight_candidates:
            if funding_raised >= funding_needed:
                break
                
            # Calculate how much to sell from this position
            sell_value = min(candidate['excess_value'], funding_needed - funding_raised)
            commission_rate = float(candidate['instrument'].taker_fee)
            
            # Account for commission: sell_proceeds = sell_value, so sell_amount = sell_value / (1 - commission)
            sell_amount_gross = sell_value / (1 - commission_rate)
            sell_qty = sell_amount_gross / candidate['current_price']
            
            # Apply precision and ensure we don't sell more than we have
            precision = candidate['instrument'].size_precision
            sell_qty = min(round(sell_qty, precision), candidate['current_qty'])
            
            if sell_qty > 1e-8:  # Only meaningful trades
                additional_sells.append({
                    'instrument': candidate['instrument'],
                    'quantity': sell_qty,
                    'side': OrderSide.SELL,
                    'target_value': 0,  # This is a funding sell, not part of normal rebalance
                    'current_value': candidate['current_qty'] * candidate['current_price'],
                    'diff_value': -sell_qty * candidate['current_price']
                })
                
                funding_raised += sell_qty * candidate['current_price'] * (1 - commission_rate)
                print(f"🔄 Additional funding sell: {candidate['instrument'].id} {sell_qty:.6f} units (${sell_value:.2f})")
        
        if funding_raised < funding_needed:
            print(f"⚠️ Could only raise ${funding_raised:.2f} of needed ${funding_needed:.2f} from overweight positions")
        
        return additional_sells

    def on_stop(self):
        self.log.info(f"{self.id}: Backtest finished.")