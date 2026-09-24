# API Strategy Pattern

Generate legal moves locally, send only legal move ids to the model, and accept only an id response. This prevents API players from making invalid board moves.

Remote calls use `MINIMAX_TIMEOUT` and `max_retries=0`; failures fall back to the best locally ranked move so simulations do not stall.
