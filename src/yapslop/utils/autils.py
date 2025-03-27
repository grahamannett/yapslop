from typing import Callable, Any


async def allow_retry(func: Callable[..., Any], *args, **kwargs):
    """
    Allow a function to be retried a certain number of times.
    prefer this over decorating design pattern since this is more straightforward

    use with partial and can then pass in the max_retries.

    Still seems like a bad pattern though because if you specify the name in the
    partial you must pass the func by name later on as well
    """
    _errs = []
    max_retries = kwargs.pop("max_retries", 3)
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as err:
            _errs.append(f"Attempt {attempt+1} failed: {err}")

    error_detail = "\n".join(_errs)
    raise ValueError(f"Function {func.__name__} failed after {max_retries} attempts:\n{error_detail}")
