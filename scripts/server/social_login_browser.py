"""Headed CloakBrowser on a virtual display so ZEKE can log into TikTok / YouTube himself via noVNC.
I never see or store his password; the session cookie persists in state/cloak_profiles/social.
Run via scripts/server/social_login.sh start|stop. READ-ONLY use afterwards (analytics only)."""
import asyncio
from cloakbrowser import launch_persistent_context_async

PROFILE = "/home/iris/IRIS/state/cloak_profiles/social"


async def main():
    ctx = await launch_persistent_context_async(PROFILE, headless=False, viewport={"width": 1280, "height": 860},
                                                locale="en-US", timezone="America/New_York",
                                                args=["--start-maximized"])
    p1 = ctx.pages[0] if ctx.pages else await ctx.new_page()
    await p1.goto("https://www.tiktok.com/login", timeout=60000)
    p2 = await ctx.new_page()
    await p2.goto("https://studio.youtube.com", timeout=60000)
    await p1.bring_to_front()
    while True:
        await asyncio.sleep(3600)

asyncio.run(main())
