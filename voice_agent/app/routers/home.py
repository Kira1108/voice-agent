"""The single entry page: a sidebar with the app's pages (Voice Agent, 声音复刻), each shown in a frame."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"

router = APIRouter(tags=["home"])


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return PAGE.read_text(encoding="utf-8")  # read per request, so edits show up on reload
