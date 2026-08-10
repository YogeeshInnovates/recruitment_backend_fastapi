import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class InternalApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        api_key = os.getenv("INTERNAL_API_KEY", "")

        if not api_key:
            # Optional auth: when INTERNAL_API_KEY is not configured (e.g. local dev),
            # requests pass through so the service keeps working without env setup.
            return await call_next(request)

        provided_key = request.headers.get("X-Internal-Api-Key", "")

        if not provided_key or provided_key != api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing internal API key"}
            )

        return await call_next(request)
