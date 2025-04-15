"""
Module untuk analisis teknikal dan sinyal trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

def generate_trading_signals(price_df: pd.DataFrame, project_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate trading signals based on technical analysis
    
    Args:
        price_df: DataFrame with price history
        project_info: Additional project information
    
    Returns:
        dict: Trading signals and analysis
    """
    # Ensure DataFrame is sorted by timestamp
    price_df = price_df.sort_values('timestamp')
    
    # Add technical indicators
    price_df = add_technical_indicators(price_df)
    
    # Calculate price trends
    price_change_24h = calculate_price_change(price_df, hours=24)
    price_change_7d = calculate_price_change(price_df, hours=168)
    price_change_1h = calculate_price_change(price_df, hours=1)
    
    # Detect volume spikes
    volume_spike = detect_volume_spike(price_df)
    
    # Basic signal logic
    signal = determine_signal(
        price_df, 
        price_change_1h,
        price_change_24h, 
        price_change_7d, 
        volume_spike
    )
    
    # Get confidence level
    confidence = calculate_confidence(signal, price_df, project_info)
    
    # Compile result
    result = {
        "action": signal["action"],
        "confidence": confidence,
        "reason": signal["reason"],
        "metrics": {
            "price_change_1h": price_change_1h,
            "price_change_24h": price_change_24h,
            "price_change_7d": price_change_7d,
            "current_price": float(price_df['price'].iloc[-1]),
            "volume_change_24h": float(volume_spike["change"]),
            "rsi_14": float(price_df['rsi_14'].iloc[-1]) if 'rsi_14' in price_df.columns else None,
            "macd": float(price_df['macd'].iloc[-1]) if 'macd' in price_df.columns else None,
            "macd_signal": float(price_df['macd_signal'].iloc[-1]) if 'macd_signal' in price_df.columns else None
        },
        "updated_at": price_df['timestamp'].max().isoformat() if 'timestamp' in price_df.columns else None,
        "signals": {
            "moving_average": signal.get("ma_signal"),
            "rsi": signal.get("rsi_signal"),
            "macd": signal.get("macd_signal"),
            "volume": signal.get("volume_signal"),
            "price_momentum": signal.get("momentum_signal")
        }
    }
    
    # Add recommended price targets if available
    if "target_price" in signal:
        result["target_price"] = signal["target_price"]
    
    # Add stop loss if it's a buy signal
    if signal["action"] == "buy" and "stop_loss" in signal:
        result["stop_loss"] = signal["stop_loss"]
    
    return result

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to price DataFrame"""
    # Copy DataFrame to avoid modifying original
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['sma_50'] = df['price'].rolling(window=50).mean()
    df['sma_200'] = df['price'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI (14-period)
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df

def calculate_price_change(df: pd.DataFrame, hours: int = 24) -> float:
    """Calculate price change over specified period"""
    if len(df) < hours:
        return 0.0
        
    current_price = df['price'].iloc[-1]
    prev_price = df['price'].iloc[-min(hours, len(df))]
    
    if prev_price == 0:
        return 0.0
        
    return ((current_price - prev_price) / prev_price) * 100

def detect_volume_spike(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect volume spikes"""
    if len(df) < 24 or 'volume' not in df.columns:
        return {"detected": False, "change": 0}
    
    # Get the last 24 hours volume
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].iloc[-24:-1].mean()
    
    if avg_volume == 0:
        return {"detected": False, "change": 0}
    
    volume_change = ((current_volume - avg_volume) / avg_volume) * 100
    
    return {
        "detected": volume_change > 50,  # 50% increase in volume
        "change": volume_change
    }

def determine_signal(df: pd.DataFrame, price_change_1h: float, price_change_24h: float, 
                     price_change_7d: float, volume_spike: Dict[str, Any]) -> Dict[str, Any]:
    """Determine trading signal based on indicators"""
    # Default response
    signal = {
        "action": "hold",
        "reason": "No clear signals detected"
    }
    
    # Skip if not enough data
    if len(df) < 50:
        return signal
    
    # Get latest values
    current_price = df['price'].iloc[-1]
    rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
    macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
    macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
    sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else current_price
    sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else current_price
    
    # Individual signals
    ma_signal = "bullish" if sma_20 > sma_50 else "bearish"
    rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    macd_signal = "bullish" if macd > macd_signal else "bearish"
    volume_signal = "high" if volume_spike["detected"] else "normal"
    momentum_signal = "positive" if price_change_24h > 0 else "negative"
    
    # Store signals
    signal["ma_signal"] = ma_signal
    signal["rsi_signal"] = rsi_signal
    signal["macd_signal"] = macd_signal
    signal["volume_signal"] = volume_signal
    signal["momentum_signal"] = momentum_signal
    
    # Buy signal conditions
    if ((rsi < 40 and macd > macd_signal) or 
        (price_change_24h < -10 and price_change_1h > 2) or 
        (rsi_signal == "oversold" and volume_spike["detected"])):
        
        signal["action"] = "buy"
        
        # Generate reason based on strongest indicator
        if rsi < 30:
            signal["reason"] = f"RSI indicates oversold condition ({rsi:.2f})"
        elif macd > macd_signal and macd_signal < 0:
            signal["reason"] = "MACD shows bullish crossover from negative territory"
        elif price_change_24h < -10:
            signal["reason"] = f"Price dip with recovery (24h: {price_change_24h:.2f}%, 1h: {price_change_1h:.2f}%)"
        else:
            signal["reason"] = "Multiple technical indicators suggest buying opportunity"
            
        # Set target price (5-10% above current price)
        signal["target_price"] = current_price * (1 + (abs(price_change_24h) / 200))
        signal["stop_loss"] = current_price * 0.95  # 5% below current price
    
    # Sell signal conditions
    elif ((rsi > 70 and macd < macd_signal) or 
          (price_change_24h > 15 and price_change_1h < -1) or 
          (rsi_signal == "overbought" and price_change_7d > 30)):
          
        signal["action"] = "sell"
        
        # Generate reason based on strongest indicator
        if rsi > 70:
            signal["reason"] = f"RSI indicates overbought condition ({rsi:.2f})"
        elif macd < macd_signal and macd_signal > 0:
            signal["reason"] = "MACD shows bearish crossover from positive territory"
        elif price_change_24h > 15:
            signal["reason"] = f"Sharp price increase followed by reversal (24h: {price_change_24h:.2f}%, 1h: {price_change_1h:.2f}%)"
        else:
            signal["reason"] = "Multiple technical indicators suggest selling opportunity"
            
        # Set target price (current price as reference)
        signal["target_price"] = current_price
    
    return signal

def calculate_confidence(signal: Dict[str, Any], df: pd.DataFrame, 
                         project_info: Optional[Dict[str, Any]] = None) -> float:
    """Calculate confidence level for the signal"""
    # Default medium confidence
    confidence = 0.5
    
    # Skip if hold recommendation
    if signal["action"] == "hold":
        return confidence
    
    # Count confirming signals
    confirming_signals = 0
    total_signals = 5  # MA, RSI, MACD, Volume, Momentum
    
    if signal["action"] == "buy":
        if signal.get("ma_signal") == "bullish": confirming_signals += 1
        if signal.get("rsi_signal") == "oversold": confirming_signals += 1
        if signal.get("macd_signal") == "bullish": confirming_signals += 1
        if signal.get("volume_signal") == "high": confirming_signals += 1
        if signal.get("momentum_signal") == "positive": confirming_signals += 1
    elif signal["action"] == "sell":
        if signal.get("ma_signal") == "bearish": confirming_signals += 1
        if signal.get("rsi_signal") == "overbought": confirming_signals += 1
        if signal.get("macd_signal") == "bearish": confirming_signals += 1
        if signal.get("volume_signal") == "high": confirming_signals += 1
        if signal.get("momentum_signal") == "negative": confirming_signals += 1
    
    # Base confidence on confirming signals ratio
    confidence = 0.3 + (0.4 * (confirming_signals / total_signals))
    
    # Adjust based on project information if available
    if project_info is not None:
        # More confidence in established projects
        if 'maturity_score' in project_info:
            maturity_bonus = project_info['maturity_score'] / 200  # 0-0.5 bonus
            confidence = min(0.95, confidence + maturity_bonus)
        
        # Less confidence in extremely volatile projects
        if 'price_change_percentage_24h' in project_info:
            volatility = abs(project_info['price_change_percentage_24h'])
            if volatility > 20:
                confidence = max(0.05, confidence - 0.1)
    
    return round(confidence, 2)

def personalize_signals(signals: Dict[str, Any], risk_tolerance: str = 'medium') -> Dict[str, Any]:
    """Personalize trading signals based on user's risk tolerance"""
    personalized = signals.copy()
    
    # Adjust confidence based on risk tolerance
    if risk_tolerance == 'low':
        # More conservative signals for low risk users
        if signals['action'] == 'buy' and signals['confidence'] < 0.7:
            personalized['action'] = 'hold'
            personalized['reason'] = f"Original signal ({signals['action']}) adjusted to HOLD due to your conservative risk profile"
            personalized['original_signal'] = signals['action']
        elif signals['action'] == 'sell' and signals['confidence'] < 0.6:
            # Low risk users should sell more readily
            personalized['confidence'] = min(0.95, signals['confidence'] + 0.1)
    
    elif risk_tolerance == 'high':
        # More aggressive signals for high risk users
        if signals['action'] == 'hold' and signals['confidence'] > 0.4:
            # Try to find a signal direction for high risk users
            if 'momentum_signal' in signals.get('signals', {}):
                if signals['signals']['momentum_signal'] == 'positive':
                    personalized['action'] = 'buy'
                    personalized['reason'] = "Momentum trending positive - opportunistic buy for aggressive risk profile"
                    personalized['confidence'] = 0.5
                elif signals['signals']['momentum_signal'] == 'negative':
                    personalized['action'] = 'sell'
                    personalized['reason'] = "Momentum trending negative - opportunistic sell for aggressive risk profile"
                    personalized['confidence'] = 0.5
    
    return personalized