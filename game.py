import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from abc import ABC, abstractmethod

# Define constants
PRICE_LB, PRICE_UB = 0.0, 10.0
SIZE_LB, SIZE_UB = 1, 100
SPREADS = [4.0, 2.0, 1.0]
EPSILON = 1e-6

class Contract:
    def __init__(self, price: float, size: int):
        self.price, self.size = price, size

    def buy(self, true_price: float) -> float:
        return (true_price - self.price) * self.size

    def sell(self, true_price: float) -> float:
        return (self.price - true_price) * self.size

class TakerDecision(Enum):
    BOUGHT = 1
    SOLD = -1

class Maker(ABC):
    @abstractmethod
    def make_market(self, spread: float) -> (Contract, Contract):
        pass

    @abstractmethod
    def receive_decision(self, decision: TakerDecision) -> None:
        pass

class Taker(ABC):
    @abstractmethod
    def take_market(self, bid: Contract, ask: Contract, true_price: float) -> (TakerDecision, float):
        pass

class GreedyTaker(Taker):
    def take_market(self, bid: Contract, ask: Contract, true_price: float) -> (TakerDecision, float):
        buy_profit = ask.buy(true_price)
        sell_profit = bid.sell(true_price)
        return (
            (TakerDecision.BOUGHT, buy_profit)
            if (buy_profit > sell_profit)
            else (TakerDecision.SOLD, sell_profit)
        )

def run_one_sim(maker: Maker = None, taker: Taker = GreedyTaker(), true_price: float = None, verbose: bool = False) -> (float, float):
    assert maker is not None, "Must give a Maker object to `run_one_sim`"
    maker_profit = 0.0
    if true_price is None:
        true_price = random.uniform(0, 10)
    if verbose:
        print(f"The true price is {round(true_price, 2)}")

    for round_number, spread in enumerate(SPREADS):
        bid, ask = maker.make_market(spread)

        if verbose:
            print(f"\nRound {round_number+1}: spread is at most {spread}")
            print(f"Maker bids with price={bid.price}, size={bid.size}")
            print(f"Maker asks with price={ask.price}, size={ask.size}")

        # Check that contracts satisfy requirements
        assert (type(bid.price) is float) and (type(ask.price) is float), "Price must be a float"
        assert (type(bid.size) is int) and (type(ask.size) is int), "Size must be an integer"
        assert (ask.price - bid.price) <= spread + EPSILON, f"Spread must be at most {spread}"
        assert (SIZE_LB <= ask.size <= SIZE_UB) and (SIZE_LB <= bid.size <= SIZE_UB), f"Contract size must be between {SIZE_LB} and {SIZE_UB}"
        assert (PRICE_LB <= ask.price <= PRICE_UB) and (PRICE_LB <= bid.price <= PRICE_UB), f"Contract price must be between {PRICE_LB} and {PRICE_UB}"

        taker_decision, taker_profit = taker.take_market(bid, ask, true_price)
        maker.receive_decision(taker_decision)
        maker_profit -= taker_profit

        if verbose:
            print(f"The taker {taker_decision.name} at {round(true_price - taker_profit * taker_decision.value, 2)}")
            print(f"Maker gained {round(-taker_profit, 2)} this round")

    return true_price, maker_profit

def run_many_sims(maker_cls: Maker, taker_cls: Taker = GreedyTaker, num_sims: int = int(1e4)) -> pd.DataFrame:
    prices, profits = [], []
    for n in range(num_sims):
        true_price, maker_profit = run_one_sim(maker_cls(), taker_cls())
        prices.append(true_price)
        profits.append(maker_profit)
    ser = pd.Series(profits, index=prices).sort_index()

    # Plot maker profits versus true price
    fig, ax = plt.subplots(figsize=(8, 4))
    ser.plot.line(ax=ax)
    ax.title.set_text("Maker Profit for Different Values of True Price")
    ax.set_xlabel("True Price")
    ax.set_ylabel("Maker Profit")
    fig.tight_layout()
    ax.axhline(0, color="g", ls=":")
    plt.xticks(np.arange(0.0, 11.0, 1.0))
    plt.grid()

    # Return a dataframe of summary statistics
    return ser.describe()