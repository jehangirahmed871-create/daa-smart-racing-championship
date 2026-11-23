"""Step-hook helper.

Every algorithm is written ONCE as a generator `<name>_steps(..., emit=True)`.
  * The Algorithm Lab iterates it and draws each yielded step dict.
  * Gameplay calls the plain wrapper, which runs the same generator with
    emit=False (no step objects are created, so timing is not distorted).
"""


def run_to_completion(gen):
    """Exhaust a step generator and return its final result."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value
