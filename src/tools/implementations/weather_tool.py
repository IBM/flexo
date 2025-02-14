# src/tools/implementations/weather_tool.py

import os
from typing import Optional, Any, Dict

from src.data_models.tools import ToolResponse
from src.data_models.agent import StreamContext
from src.tools.core.tool_registry import ToolRegistry
from src.utils.json_formatter import format_json_to_document
from src.tools.core.base_rest_tool import BaseRESTTool, ResponseFormat


@ToolRegistry.register_tool()
class WeatherTool(BaseRESTTool):
    name = "weather_tool"

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config=config)
        self.description = 'Get current temperature and weather information for a specified location.'
        self.strict = False

        self.parameters = {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'City name (e.g., "London", "New York")'
                },
                'state_code': {
                    'type': 'string',
                    'description': 'US state code (e.g., "NY", "CA"). Only applicable for US locations.'
                },
                'country_code': {
                    'type': 'string',
                    'description': 'Two-letter country code (e.g., "US", "GB"). Optional, helps disambiguate cities.'
                },
                'units': {
                    'type': 'string',
                    'description': 'Temperature units: Use "metric" for Celsius, "imperial" for Fahrenheit, or "standard" for Kelvin. Defaults to metric.'
                },
                'lang': {
                    'type': 'string',
                    'description': 'Language for weather descriptions (e.g., "en", "es", "fr"). Defaults to "en".'
                }
            },
            'required': ['city'],
            'additionalProperties': False
        }

        self.content_type = "application/json"
        self.rate_limit = 60
        self.default_timeout = 10
        self.max_retries = 3
        self.retry_delay = 1.0

    async def execute(self, context: Optional[StreamContext] = None, **kwargs) -> ToolResponse:
        """Execute a weather data request.

        Args:
            context (Optional[Dict]): Additional context for the request.
            **kwargs: Must include:

                - city (str): City name
                Optional:
                - state_code (str): US state code
                - country_code (str): Two-letter country code
                - units (str): 'metric', 'imperial', or 'standard'
                - lang (str): Language code

        Returns:
            str: Formatted weather information.

        Raises:
            ValueError: If required parameters or API key are missing.
        """
        # Extract and validate parameters
        city = kwargs.get('city')
        if not city:
            raise ValueError("City parameter is required")

        # Get API key from environment
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")

        state_code = kwargs.get('state_code')
        country_code = kwargs.get('country_code')
        units = kwargs.get('units', 'metric')
        lang = kwargs.get('lang', 'en')

        # Construct the location query
        location_parts = [city]
        if state_code and country_code == 'US':
            location_parts.append(state_code)
        if country_code:
            location_parts.append(country_code)

        location = ','.join(location_parts)

        # Prepare query parameters
        query_params = {
            'q': location,
            'units': units,
            'lang': lang,
            'appid': api_key  # OpenWeatherMap expects the API key as 'appid'
        }

        # Log the request details
        self.logger.info(f"Making weather request for location: {location}")
        self.logger.info(f"Query parameters: {query_params}")
        self.logger.debug(f"Context: {context}")

        # Make the API request using parent class method
        response = await self.make_request(
            method="GET",
            params=query_params,
            endpoint_url=self.endpoint,
            response_format=ResponseFormat.JSON,
            use_token=False,  # OpenWeatherMap uses an API key
            additional_headers={
                "Accept": "application/json",
                "User-Agent": "WeatherTool/1.0"
            }
        )
        response = ToolResponse(
            result=self.parse_output(response),
            context=None,
        )
        return response

    def parse_output(self, output: Any) -> str:
        """Parse and format the weather API response.

        Args:
            output (Any): Raw API response data.

        Returns:
            str: Formatted weather information including location,
                temperature, humidity, wind, etc.

        Note:
            Handles error responses and includes unit-specific information
            in the output.
        """
        try:
            if not isinstance(output, dict):
                return str(output)

            if 'error' in output:
                return f"Error: {output['error']}"

            if output.get('cod') != 200:
                error_msg = output.get('message', 'Unknown error')
                return f"Error: {error_msg}"

            # Format the weather data into a more readable structure
            weather_info = {
                'location': {
                    'city': output.get('name'),
                    'country': output.get('sys', {}).get('country'),
                    'coordinates': {
                        'latitude': output.get('coord', {}).get('lat'),
                        'longitude': output.get('coord', {}).get('lon')
                    }
                },
                'current_weather': {
                    'temperature': output.get('main', {}).get('temp'),
                    'feels_like': output.get('main', {}).get('feels_like'),
                    'humidity': output.get('main', {}).get('humidity'),
                    'pressure': output.get('main', {}).get('pressure'),
                    'description': output.get('weather', [{}])[0].get('description'),
                    'main': output.get('weather', [{}])[0].get('main'),
                    'wind': {
                        'speed': output.get('wind', {}).get('speed'),
                        'direction': output.get('wind', {}).get('deg')
                    },
                    'clouds': output.get('clouds', {}).get('all'),
                    'visibility': output.get('visibility')
                }
            }

            formatted_output = format_json_to_document(weather_info)
            formatted_output += self.get_tool_specific_instruction()

            return formatted_output

        except Exception as e:
            self.logger.error(f"Failed to parse weather data: {e}", exc_info=True)
            return "An error occurred while parsing the weather information."

    def get_tool_specific_instruction(self) -> str:
        """Get tool-specific instructions about weather data units.

        Returns:
            str: Instructions about weather data units and measurements.
        """
        return (
            "\n\n"
            "## Weather Information Notes: ##\n"
            "- Temperature and feels_like are in the requested units (Celsius for metric, Fahrenheit for imperial)\n"
            "- Wind speed is in meters/sec for metric, miles/hour for imperial\n"
            "- Visibility is in meters\n"
            "- Pressure is in hPa (hectopascals)"
        )
