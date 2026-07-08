"""Google Maps search-link helpers for specialist discovery.

No Google Maps API key or billing is used. The frontend opens the generated
search URL in the user's browser.
"""
from urllib.parse import quote_plus


def maps_search_url(specialist: str = "General Physician") -> str:
    specialist = (specialist or "General Physician").strip()
    return f"https://www.google.com/maps/search/{quote_plus(specialist + ' near me')}"
