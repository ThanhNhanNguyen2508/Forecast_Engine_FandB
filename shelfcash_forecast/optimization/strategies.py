from shelfcash_forecast.optimization.contracts import StrategyProfile


def default_strategy_profiles() -> list[StrategyProfile]:
    """Explicit product defaults; callers may replace every coefficient."""

    return [
        StrategyProfile(
            name="LEAN",
            shortage_penalty=1,
            holding_penalty=3,
            waste_penalty=3,
            cash_penalty=1,
            cvar_weight=0,
            minimum_acceptable_fill_rate=0.80,
            maximum_acceptable_stockout_probability=0.35,
        ),
        StrategyProfile(
            name="BALANCED",
            shortage_penalty=10,
            holding_penalty=1,
            waste_penalty=1,
            cash_penalty=0.2,
            cvar_weight=0.25,
            minimum_acceptable_fill_rate=0.90,
            maximum_acceptable_stockout_probability=0.15,
        ),
        StrategyProfile(
            name="PROTECTED",
            shortage_penalty=50,
            holding_penalty=0.2,
            waste_penalty=0.5,
            cash_penalty=0.05,
            cvar_weight=1,
            maximum_stockout_probability=0.05,
            minimum_expected_fill_rate=0.95,
            minimum_acceptable_fill_rate=0.95,
            maximum_acceptable_stockout_probability=0.05,
        ),
    ]
