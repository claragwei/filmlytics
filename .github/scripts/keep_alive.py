"""Open the deployed Streamlit app in a headless browser so it stays awake.

A plain HTTP request does NOT wake a sleeping Streamlit Community Cloud app; a real
browser session does. Exits non-zero if the app never renders, so the Action turns red.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]
MARKER = "Go to"          # sidebar radio label that only appears once the app has rendered
TIMEOUT_S = 240           # waking a sleeping app can take a few minutes


def app_rendered(page) -> bool:
    return any(MARKER in f.content() for f in page.frames)


def click_wake_button(page) -> bool:
    for f in page.frames:
        btn = f.get_by_role("button", name="Yes, get this app back up!")
        if btn.count():
            btn.first.click()
            return True
    return False


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    deadline = time.time() + TIMEOUT_S
    woke = False
    while time.time() < deadline:
        if app_rendered(page):
            print(f"OK: app rendered{' (was asleep, woke it)' if woke else ''}.")
            browser.close()
            sys.exit(0)
        if not woke and click_wake_button(page):
            woke = True
            print("App was asleep; clicked the wake-up button.")
        page.wait_for_timeout(3000)
    page.screenshot(path="keep_alive_failure.png")
    print(f"FAIL: app did not render within {TIMEOUT_S}s (is it public? did the deploy fail?)")
    browser.close()
    sys.exit(1)
