from __future__ import annotations

import httpx
from pydantic import BaseModel

API_URL = "https://api.exchangerate.host/latest"


class RateResponse(BaseModel):
    base: str
    rates: dict[str, float]


async def get_rates(base: str = "USD") -> RateResponse:
    params = {"base": base}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(API_URL, params=params)
        r.raise_for_status()
        data = r.json()
        return RateResponse(base=data.get("base", base), rates=data.get("rates", {}))


def convert(amount: float, from_code: str, to_code: str, rates: RateResponse) -> float:
    if from_code == rates.base:
        rate = rates.rates.get(to_code)
        if rate is None:
            raise ValueError(f"No rate for {to_code}")
        return amount * rate

    # Convert via base
    to_base = rates.rates.get(from_code)
    if to_base is None or to_base == 0:
        raise ValueError(f"No rate for {from_code}")
    amount_base = amount / to_base

    if to_code == rates.base:
        return amount_base

    out_rate = rates.rates.get(to_code)
    if out_rate is None:
        raise ValueError(f"No rate for {to_code}")
    return amount_base * out_rate
