"""Weather forecasts from Open-Meteo (free, no key). Try it:   python -m tools.weather"""

import requests


def get_weather_forecast(place, region=""):
    """The daily forecast for the next 7 days, as text."""
    try:
        found = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 10}, timeout=30).json()
        if "results" not in found:
            return f"Could not find a place called {place}."
        chosen = found["results"][0]
        for candidate in found["results"]:
            if region != "" and region.lower() in candidate.get("admin1", "").lower():
                chosen = candidate
                break
        daily = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": chosen["latitude"], "longitude": chosen["longitude"], "timezone": "auto", "forecast_days": 7,
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum"}, timeout=30).json()["daily"]
    except requests.RequestException as error:
        return f"The weather service is unavailable right now ({type(error).__name__})."
    lines = []
    for position in range(len(daily["time"])):
        lines.append(f"{daily['time'][position]}: {daily['temperature_2m_min'][position]} to "
                     f"{daily['temperature_2m_max'][position]} C, rain {daily['precipitation_sum'][position]} mm")
    return f"7-day forecast for {chosen['name']}, {chosen.get('admin1', '')} (covers only these dates): " + "; ".join(lines)


if __name__ == "__main__":
    print(get_weather_forecast("Chandigarh"))