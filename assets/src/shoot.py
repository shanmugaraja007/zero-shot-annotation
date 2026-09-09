from playwright.sync_api import sync_playwright
import pathlib, sys
N=int(sys.argv[1]) if len(sys.argv)>1 else 120
out=pathlib.Path("frames"); out.mkdir(exist_ok=True)
for f in out.glob("*.png"): f.unlink()
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":1660,"height":800}, device_scale_factor=1)
    pg=ctx.new_page()
    pg.goto("file:///home/claude/zsa/assets/src/diagram.html", wait_until="networkidle")
    pg.wait_for_timeout(600)
    imgs=pg.evaluate("[...document.images].map(i=>[i.alt,i.naturalWidth])")
    print("images:",imgs)
    for i in range(N):
        pg.evaluate(f"window.setTime({i/N})")
        pg.screenshot(path=str(out/f"f{i:04d}.png"))
    b.close()
print("frames:",N)
