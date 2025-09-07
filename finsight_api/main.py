# finsight_api/main.py

from fastapi import FastAPI
from finsight_api.routers import stocks

# Create the main FastAPI application instance
app = FastAPI(
    title="FinSight API",
    description="A financial analysis engine using ML, LLMs, and real-time data to provide actionable stock signals.",
    version="1.0.0",
)

# Include the stocks router
# This makes all the endpoints defined in stocks.py available in our app
app.include_router(stocks.router)

# Define a root endpoint for health checks
@app.get("/", tags=["Root"])
async def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "Welcome to the FinSight API!"}