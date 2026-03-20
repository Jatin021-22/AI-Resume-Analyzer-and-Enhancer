from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.get("/status")
async def auth_status():
    return {"status": "ok", "message": "Auth module ready — JWT coming soon"}
