import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.auth import InternalApiKeyMiddleware
from routes import ai_routes, interview_routes

app = FastAPI(
    title="Recruitment AI Service",
    description="AI-powered recruitment assistant",
    version="1.0.0"
)

app.add_middleware(InternalApiKeyMiddleware)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_routes.router, prefix="/api/ai", tags=["AI"])
app.include_router(interview_routes.router, prefix="/api/ai/interview", tags=["Interview"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "recruitment-ai-service"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
