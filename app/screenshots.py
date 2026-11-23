"""Take three screenshots of the demo with a headless browser (Playwright / Chromium).

    python app/screenshots.py --out figures --prefix demo_synthetic   # whatever artefacts the app finds
    python app/screenshots.py --out artifacts/screenshots --prefix demo_real

Starts Streamlit on a free local port, drives the page, saves PNGs, stops the server.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESS = ("Example Corp (NYSE: EXM) today announced that it has entered into a definitive merger agreement to be acquired by an "
         "investment fund affiliated with a private equity firm for $42.00 per share in cash, a premium of 35%. The transaction is "
         "not subject to a financing condition; the buyer has obtained debt and equity commitment letters. Closing is subject to "
         "approval by Example's stockholders, expiration of the waiting period under the Hart-Scott-Rodino Act and other regulatory "
         "approvals. The agreement includes a 35-day go-shop period and a reverse termination fee payable by the buyer.")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures"); ap.add_argument("--prefix", default="demo")
    ap.add_argument("--force-synthetic", action="store_true", help="hide the real artefacts for this run")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    real = ROOT / "artifacts/demo/demo.pkl"; hidden = real.with_suffix(".pkl.hidden")
    if args.force_synthetic and real.exists():
        real.rename(hidden)
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app/demo.py"), "--server.headless", "true",
                            "--server.port", str(port), "--server.address", "127.0.0.1"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1500, "height": 1000})
            for _ in range(60):
                try:
                    pg.goto(f"http://127.0.0.1:{port}", timeout=5000); break
                except Exception:
                    time.sleep(1)
            pg.wait_for_selector("text=Model break probability", timeout=120000)
            time.sleep(4)
            pg.screenshot(path=str(out / f"{args.prefix}_1_deal.png"), full_page=True)
            pg.get_by_text("Reveal what happened").click()
            time.sleep(3)
            pg.screenshot(path=str(out / f"{args.prefix}_2_outcome.png"), full_page=True)
            pg.get_by_text("Enter your own deal").click()
            pg.wait_for_selector("text=Paste the press release", timeout=60000)
            pg.locator("textarea").first.fill(PRESS)
            pg.get_by_text("Financial buyer", exact=True).click()
            pg.get_by_text("Leveraged buyout", exact=True).click()
            pg.get_by_text("Go-shop", exact=True).click()
            pg.keyboard.press("Tab")
            time.sleep(5)
            pg.screenshot(path=str(out / f"{args.prefix}_3_own_deal.png"), full_page=True)
            b.close()
    finally:
        srv.terminate()
        if hidden.exists():
            hidden.rename(real)
    print("saved:", sorted(q.name for q in out.glob(f"{args.prefix}_*.png")))


if __name__ == "__main__":
    main()
