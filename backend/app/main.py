from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(router, prefix="/api/v1", tags=["api"])
app.mount(
    "/frontend",
    StaticFiles(directory=settings.frontend_dir, html=True),
    name="frontend",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": f"{settings.app_name} is running",
        "docs": "/docs",
        "frontend": "/frontend",
    }
