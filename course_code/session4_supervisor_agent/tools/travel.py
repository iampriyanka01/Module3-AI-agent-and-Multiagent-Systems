"""Travel guide text from Wikivoyage (free, no key). Try it:   python -m tools.travel_guide"""

import requests


def search_travel_guide(place, max_characters=4000):
    try:
        response = requests.get("https://en.wikivoyage.org/w/api.php",
                                params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                        "titles": place, "format": "json"},
                                headers={"User-Agent": "DilSeTrips-course-project/1.0 (educational use)"}, timeout=30)
        pages = response.json()["query"]["pages"]
    except requests.RequestException as error:
        return f"The travel guide is unavailable right now ({type(error).__name__})."
    for page_id in pages:
        if "extract" in pages[page_id]:
            return pages[page_id]["extract"][:max_characters]
    return f"No Wikivoyage page found for {place}."


if __name__ == "__main__":
    print(search_travel_guide("Chandigarh")[:800])