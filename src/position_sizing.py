def get_position_size(confidence, regime, base_size, confidence_threshold=0.6, regime_map=None):
    """
    Determines position size based on confidence and regime.
    Args:
        confidence (float): Predicted win probability from meta-label model.
        regime (int): Cluster label from regime model.
        base_size (int): Default position size.
        confidence_threshold (float): Minimum confidence to consider trading.
        regime_map (dict, optional): Mapping from regime label to size multiplier.
                                     e.g., {0: 1.0, 1: 0.5, 2: 0.0} for Good, Neutral, Bad regimes.
    Returns:
        int: The calculated position size (can be 0).
    """
    if confidence < confidence_threshold:
        return 0 # Don't trade if confidence is too low

    size_multiplier = 1.0 # Default multiplier

    if regime_map and regime in regime_map:
        size_multiplier = regime_map[regime]

    # Optional: Scale by confidence (simple linear scaling above threshold)
    # scale_factor = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
    # size = round(base_size * size_multiplier * scale_factor)

    # Simpler: Use multiplier directly
    size = round(base_size * size_multiplier)

    # Ensure size is not negative and potentially cap it
    return max(0, int(size))

# Example usage:
# Define how your regimes affect size (you need to analyze clusters first)
# Assume regime 0 is good (trend/vol), 1 is okay, 2 is bad (choppy)
# my_regime_map = {0: 1.0, 1: 0.5, 2: 0.0}
# size = get_position_size(confidence=0.75, regime=0, base_size=5, regime_map=my_regime_map)