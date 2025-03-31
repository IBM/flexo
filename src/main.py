"""
Main module for initializing and configuring the FastAPI application.

This module sets up logging, loads environment variables, configures the FastAPI app,
and registers routes and middleware. It also defines global exception handlers to provide
consistent error responses.

Example:
    To run the application:
        $ uvicorn src.main:app --reload

Environment Variables:
    LOG_LEVEL (str): Logging level for the application (default: 'INFO').
"""

import os
import yaml
import logging
from fastapi import FastAPI
from starlette import status
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.api.sse_models import SSEChunk
from src.api.routes.chat_completions_api import router as chat_completions_router

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Initializing application with log level: %s", log_level)

# Initialize FastAPI app
app = FastAPI(
    title="Chat Completions API",
    description="API for handling chat completions with streaming support",
    version="1.0.0",
)

# Load agent config to check for allowed_origins
try:
    with open("src/configs/agent.yaml", "r") as file:
        agent_config = yaml.safe_load(file)
except Exception as e:
    logger.warning("Failed to load agent configuration for CORS: %s", str(e))
    agent_config = {}

allowed_origins = agent_config.get("allowed_origins")
if allowed_origins:
    # If allowed_origins is provided as a string, convert it to a list.
    if isinstance(allowed_origins, str):
        allowed_origins = [allowed_origins]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-KEY"],
    )
    logger.info("CORS middleware enabled with allowed origins: %s", allowed_origins)
else:
    logger.info("CORS middleware not enabled because no allowed_origins were provided in the config.")

# Include the chat completions router
logger.info("Registering chat completions router at /v1")
app.include_router(chat_completions_router, prefix="/v1")


# --------------------
# EXCEPTION HANDLERS
# --------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    """Handle all unhandled exceptions globally.

    This function catches any exception that is not explicitly handled by other exception
    handlers and returns a JSON response with the error details.

    Args:
        request (Request): The incoming request.
        exc (Exception): The exception that was raised.

    Returns:
        JSONResponse: A JSON response with an appropriate HTTP status code and error message.
    """
    error_msg = str(exc)
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    if isinstance(exc, ValueError):
        logger.warning("ValueError occurred: %s", error_msg)
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, TimeoutError):
        logger.error("Request timed out: %s", error_msg)
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
        error_msg = "Request timed out. Please try again."
    else:
        logger.error("Unhandled exception occurred", exc_info=True)

    logger.debug("Generating error response chunk with status %d", status_code)
    error_response = await SSEChunk.stop_event(
        refusal=error_msg,
        content="I apologize, but an error occurred while processing your request.",
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Handle request validation errors.

    This function catches errors that occur during request validation (e.g., invalid format,
    missing required fields) and returns a JSON response with details about the validation failure.

    Args:
        request (Request): The incoming request.
        exc (RequestValidationError): The exception raised during request validation.

    Returns:
        JSONResponse: A JSON response with a 422 status code and details about the validation errors.
    """
    logger.warning("Request validation failed: %s", str(exc))
    logger.debug("Validation error details: %s", exc.errors())

    error_response = await SSEChunk.stop_event(
        refusal=str(exc),
        content="There seems to be an issue with the provided request format. Please check your input and try again.",
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )
