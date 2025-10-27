# Minimum position size to trigger position merging
# Positions smaller than this will be ignored to save on gas costs
MIN_MERGE_SIZE = 20

# _POSITION_EPS defines the absolute minimum position size (in YES shares) that we
# still treat as meaningfully non-zero. Anything at or below this threshold is
# considered flat for liquidation checks, forced exits, and buy-side gating.
# Raise the value if tiny residual fills keep triggering forced exits; lower it
# when trading very thin ticks where you need to react to sub-nano inventory.
_POSITION_EPS = 1e-9
