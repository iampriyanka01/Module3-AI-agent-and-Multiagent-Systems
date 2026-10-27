"""Tools: plain Python functions that fetch real information. No AI in this file.

- Weather (Open-Meteo, free, no key) and the travel guide (Wikivoyage, free, no key)
- Airports, flights and hotels from LiteAPI's free sandbox (one key)

Each function returns text, so an agent can read the result directly.
Try it:   python tools.py
"""

from datetime import date, timedelta

import requests

from config import LITEAPI_API_KEY

LITEAPI = "https://api.liteapi.travel/v3.0"


# ---------------- weather and travel guide ----------------

def get_weather_forecast(place: str) -> str:
    """Get the weather forecast for the next 7 days for a town, for example Chandigarh.

    Args:
        place: The town or city.
    """
    found = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1}).json()
    if "results" not in found:
        return f"Could not find a place called {place}."
    town = found["results"][0]
    forecast = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": town["latitude"], "longitude": town["longitude"], "timezone": "auto",
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum"}).json()["daily"]
    lines = []
    for day in range(len(forecast["time"])):
        lines.append(f"{forecast['time'][day]}: {forecast['temperature_2m_min'][day]} to "
                     f"{forecast['temperature_2m_max'][day]} C, rain {forecast['precipitation_sum'][day]} mm")
    return f"7-day forecast for {town['name']} (it covers only these dates): " + "; ".join(lines)


def search_travel_guide(place: str) -> str:
    """Read the Wikivoyage travel guide for a place: what to see, do and eat.

    Args:
        place: The name of the place, for example Chandigarh.
    """
    response = requests.get("https://en.wikivoyage.org/w/api.php",
                            params={"action": "query", "prop": "extracts", "explaintext": 1,
                                    "redirects": 1, "titles": place, "format": "json"},
                            headers={"User-Agent": "DilSeTrips-course-project/1.0"})
    pages = response.json()["query"]["pages"]
    for page_id in pages:
        if "extract" in pages[page_id]:
            return pages[page_id]["extract"][:3000]
    return f"No travel guide found for {place}."


# ---------------- LiteAPI: one small helper for every request ----------------

def liteapi(method, path, params=None, body=None):
    """Send one request to LiteAPI. Returns the JSON answer, or a dictionary with an "error" sentence."""
    response = requests.request(method, LITEAPI + path, headers={"X-API-Key": LITEAPI_API_KEY},
                                params=params, json=body, timeout=90)
    data = response.json()
    if response.status_code >= 400:
        details = data.get("error", {})
        if not isinstance(details, dict):
            details = {"message": str(details)}
        return {"error": f"LiteAPI error: {details.get('description') or details.get('message') or response.status_code}"}
    return data


# ---------------- airports ----------------

def find_airport(city: str) -> str:
    """Find the airport code for a city, for example Chandigarh -> IXC.

    Args:
        city: A city or airport name.
    """
    data = liteapi("GET", "/data/iataCodes")
    if "error" in data:
        return data["error"]
    lines = []
    for airport in data.get("data", []):
        code = str(airport.get("code") or airport.get("iata") or "")
        name = str(airport.get("name") or "")
        if city.lower() == code.lower() or city.lower() in name.lower():
            lines.append(f"{code}: {name}")
        if len(lines) == 5:
            break
    if len(lines) == 0:
        return f"No airport found for {city}. Try a nearby larger city."
    return "\n".join(lines)


# ---------------- flights ----------------

def describe_flight(journey):
    """Turn one flight from LiteAPI into a dictionary of the facts a traveller cares about."""
    offer = journey["cheapestOffer"]
    segments = journey["segments"]
    numbers = []
    for segment in segments:
        numbers.append(segment["carrier"]["marketingCode"] + " " + segment["flight"]["marketingNumber"])
    checked_bag = "no checked bag"
    for bag in offer.get("baggage", {}).get("included", []):
        if bag["bagType"] == "checked" and bag["pieces"] > 0:
            checked_bag = f"{bag['weightKg']} kg checked bag included"
    return {
        "airline": segments[0]["carrier"]["marketingName"],
        "flight_numbers": ", ".join(numbers),
        "departs": segments[0]["departureTime"],
        "arrives": segments[-1]["arrivalTime"],
        "stops": len(segments) - 1,
        "minutes": journey["totalDuration"]["minutes"],
        "price": offer["pricing"]["display"]["total"],
        "fare": offer.get("fare", {}).get("family", ""),
        "checked_bag": checked_bag,
        "seats": offer.get("seats", {}).get("seatReservation", {}).get("description", ""),
    }


def search_flights(origin: str, destination: str, travel_date: str, adults: int = 2,
                   direct_only: bool = False, earliest_hour: int = 0, latest_hour: int = 23) -> str:
    """Search one-way flights. Prices are in rupees, for all travellers together.

    Args:
        origin: Airport code to fly from, for example BOM.
        destination: Airport code to fly to, for example IXC.
        travel_date: Travel date as YYYY-MM-DD.
        adults: Number of travellers.
        direct_only: True to show only direct flights.
        earliest_hour: Earliest departure hour, 0 to 23.
        latest_hour: Latest departure hour, 0 to 23.
    """
    body = {"legs": [{"origin": origin, "destination": destination, "date": travel_date}], "adults": adults, "currency": "INR"}
    data = liteapi("POST", "/flights/rates", body=body)
    if "error" in data:
        return data["error"]
    flights = []
    for block in data.get("data", []):
        for journey in block.get("journeys", []):
            flight = describe_flight(journey)
            hour = int(flight["departs"][11:13])            # "2026-11-25T09:35:00" -> 9
            if direct_only and flight["stops"] > 0:
                continue
            if hour < earliest_hour or hour > latest_hour:
                continue
            flights.append(flight)
    if len(flights) == 0:
        return "No flights match. Try relaxing a filter, or another date."
    lines = [f"{len(flights)} flights match (showing up to 8):"]
    for flight in flights[:8]:
        lines.append(f"- {flight['airline']} {flight['flight_numbers']}: departs {flight['departs']}, arrives {flight['arrives']}, "
                     f"{flight['stops']} stop(s), {flight['minutes']} min, Rs {flight['price']} total, "
                     f"{flight['fare']} fare, {flight['checked_bag']}, seats: {flight['seats']}")
    return "\n".join(lines)


# ---------------- hotels ----------------

def find_hotels(city: str, check_in: str, nights: int, adults: int = 2) -> str:
    """Find hotels with free rooms for a stay. Prices are for the whole stay, in rupees.

    Args:
        city: The city to stay in, for example Chandigarh. For Goa, use a town such as Panaji.
        check_in: Check-in date as YYYY-MM-DD.
        nights: Number of nights.
        adults: Number of guests.
    """
    # Step 1: the list of hotels in the city (real names, star ratings and review scores)
    hotel_list = liteapi("GET", "/data/hotels", params={"countryCode": "IN", "cityName": city, "limit": 15})
    if "error" in hotel_list:
        return hotel_list["error"]
    hotels = {}
    for hotel in hotel_list.get("data", []):
        hotels[hotel["id"]] = hotel
    if len(hotels) == 0:
        return f"No hotels found in {city}. Try the main town nearby."

    # Step 2: room prices for those hotels and these dates
    check_out = (date.fromisoformat(check_in) + timedelta(days=nights)).isoformat()
    rates = liteapi("POST", "/hotels/rates", body={"hotelIds": list(hotels.keys()), "occupancies": [{"adults": adults}],
                                                    "currency": "INR", "guestNationality": "IN",
                                                    "checkin": check_in, "checkout": check_out})
    if "error" in rates:
        return rates["error"]

    # Step 3: for each hotel, keep its cheapest room, and add the hotel's name from step 1
    lines = []
    for hotel_rates in rates.get("data", []):
        hotel = hotels.get(hotel_rates["hotelId"], {})
        cheapest = None
        for room in hotel_rates.get("roomTypes", []):
            if cheapest is None or room["offerRetailRate"]["amount"] < cheapest["offerRetailRate"]["amount"]:
                cheapest = room
        if cheapest is None:
            continue
        rate = cheapest["rates"][0]
        lines.append(f"- {hotel.get('name')}: {hotel.get('stars')} stars, review score {hotel.get('rating')} "
                     f"({hotel.get('reviewCount')} reviews), {hotel.get('address')}. Cheapest room: {rate['name']}, "
                     f"{rate['boardName']}, Rs {cheapest['offerRetailRate']['amount']} for {nights} nights, "
                     f"{rate['cancellationPolicies']['refundableTag']}")
        if len(lines) == 6:
            break
    if len(lines) == 0:
        return f"No free rooms in {city} for those dates."
    return "Hotels with free rooms (RFN = refundable, NRFN = non-refundable):\n" + "\n".join(lines)


if __name__ == "__main__":
    travel_day = (date.today() + timedelta(days=45)).isoformat()
    print(get_weather_forecast("Chandigarh")[:150], "\n")
    print(search_travel_guide("Chandigarh")[:150], "\n")
    print(find_airport("Chandigarh"), "\n")
    print(search_flights("BOM", "IXC", travel_day), "\n")
    print(find_hotels("Chandigarh", travel_day, 3))
