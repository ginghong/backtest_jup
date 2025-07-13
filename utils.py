import pandas as pd
from nautilus_trader.model.data import BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.adapters.binance import BinanceBar
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Money, Price, Quantity, Currency
from decimal import Decimal

def make_binance_bars_from_csv(file_path, instrument_id, price_precision, size_precision):
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.read_csv(file_path, header=None, names=columns)

    # dynamic assign 'us' or 'ms' based on opentime digits   
    if len(str(df['open_time'][0])) == 13:
        unit = 'ms'
    else:
        unit = 'us'
    df['timestamp'] = pd.to_datetime(df['open_time'], unit=unit)

    bars = []
    for _, row in df.iterrows():
        bars.append(
            BinanceBar(
                bar_type=BarType.from_str(f"{instrument_id}-1-HOUR-LAST-EXTERNAL"),
                open=Price(float(row['open']), price_precision),
                high=Price(float(row['high']), price_precision),
                low=Price(float(row['low']), price_precision),
                close=Price(float(row['close']), price_precision),
                volume=Quantity(float(row['volume']), size_precision),
                quote_volume=Quantity(float(row['quote_asset_volume']), size_precision),
                taker_buy_base_volume=Quantity(float(row['taker_buy_base_asset_volume']), size_precision),
                taker_buy_quote_volume=Quantity(float(row['taker_buy_quote_asset_volume']), size_precision),
                count=int(row['number_of_trades']),
                ts_event=dt_to_unix_nanos(row['timestamp']),
                ts_init=dt_to_unix_nanos(row['timestamp']),
            )
        )
    return bars


def make_crypto_instrument(
    symbol: str, 
    price_precision: int, 
    size_precision: int,
    venue: str = "BINANCE",
    maker_fee: float = 0.001,
    taker_fee: float = 0.001,
    min_notional: float = 10.0,
    max_quantity: float = None,
    min_quantity: float = None,
    max_price: float = None,
    min_price: float = None
) -> CurrencyPair:
    """
    Helper function to create a crypto instrument with adaptive parameters.
    
    Args:
        symbol: Trading pair symbol in format "BASE-QUOTE" (e.g., "BTC-USDT")
        price_precision: Number of decimal places for price
        size_precision: Number of decimal places for size/quantity
        venue: Trading venue name (default: "BINANCE")
        maker_fee: Maker fee as decimal (default: 0.001 = 0.1%)
        taker_fee: Taker fee as decimal (default: 0.001 = 0.1%) 
        min_notional: Minimum notional value (default: 10.0)
        max_quantity: Maximum quantity (default: calculated from precision)
        min_quantity: Minimum quantity (default: calculated from precision)
        max_price: Maximum price (default: calculated from precision)
        min_price: Minimum price (default: calculated from precision)
    
    Returns:
        CurrencyPair: Configured instrument
    """
    base, quote = symbol.split("-")
    
    # Calculate increments based on precision
    price_increment_value = 10 ** (-price_precision)
    size_increment_value = 10 ** (-size_precision)
    
    # Set adaptive defaults based on precision if not provided
    if max_quantity is None:
        max_quantity = 10 ** (10 - size_precision)  # Large but reasonable max
    if min_quantity is None:
        min_quantity = size_increment_value  # Minimum is the smallest increment
    if max_price is None:
        max_price = 10 ** (8 - price_precision)  # Large but reasonable max
    if min_price is None:
        min_price = price_increment_value  # Minimum is the smallest increment
    
    return CurrencyPair(
            instrument_id=InstrumentId(
                symbol=Symbol(f"{base}{quote}"),
                venue=Venue(venue),
            ),
            raw_symbol=Symbol(f"{base}{quote}"),
            base_currency=Currency.from_str(base),
            quote_currency=Currency.from_str(quote),
            price_precision=price_precision,
            size_precision=size_precision,
            price_increment=Price(price_increment_value, precision=price_precision),
            size_increment=Quantity(size_increment_value, precision=size_precision),
            lot_size=None,
            max_quantity=Quantity(max_quantity, precision=size_precision),
            min_quantity=Quantity(min_quantity, precision=size_precision),
            max_notional=None,
            min_notional=Money(min_notional, Currency.from_str(quote)),
            max_price=Price(max_price, precision=price_precision),
            min_price=Price(min_price, precision=price_precision),
            margin_init=Decimal(0),
            margin_maint=Decimal(0),
            maker_fee=Decimal(str(maker_fee)),
            taker_fee=Decimal(str(taker_fee)),
            ts_event=0,
            ts_init=0,
        )
