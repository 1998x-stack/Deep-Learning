#!/usr/bin/env python3
"""Chromium QA: responsive widths, MathML, SVG loading and interactive demos."""
import argparse,json,threading
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]; SITE=ROOT/"site"
SLUGS=["xor","gradient","cost-functions","output-units","hidden-units","relu","sigmoid-tanh","other-activations"]
def main():
 parser=argparse.ArgumentParser(); parser.add_argument("--output",default="reports/browser-qa"); args=parser.parse_args()
 out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
 server=ThreadingHTTPServer(("127.0.0.1",8765),partial(SimpleHTTPRequestHandler,directory=str(SITE)))
 threading.Thread(target=server.serve_forever,daemon=True).start(); report={"status":"passed","articles":{},"widths":[1440,768,390,360]}
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch()
   for slug in SLUGS:
    page=browser.new_page(viewport={"width":1440,"height":1000}); errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.on("console",lambda m:errors.append(m.text) if m.type=="error" else None)
    page.goto(f"http://127.0.0.1:8765/articles/{slug}/",wait_until="networkidle")
    result={"svg_count":page.locator("figure img[src$='.svg']").count(),"widths":{}}
    page.locator("figure img[src$='.svg']").evaluate_all("xs=>xs.forEach(x=>x.loading='eager')")
    page.wait_for_timeout(100)
    for width in report["widths"]:
     page.set_viewport_size({"width":width,"height":900}); page.wait_for_timeout(60)
     state=page.evaluate("({overflow:document.documentElement.scrollWidth>innerWidth,math:[...document.querySelectorAll('math')].every(x=>x.namespaceURI==='http://www.w3.org/1998/Math/MathML')})")
     result["widths"][str(width)]=state
     if state["overflow"] or not state["math"]: report["status"]="failed"
    broken=page.locator("figure img[src$='.svg']").evaluate_all("xs=>xs.filter(x=>!x.complete||x.naturalWidth===0).map(x=>x.src)")
    before=page.locator("[data-demo-output]").inner_text(); control=page.locator("[data-demo-input]").first
    if control.evaluate("x=>x.tagName")=="SELECT": control.select_option(index=1)
    else: control.evaluate("x=>{x.value=String(Number(x.value)+Number(x.step||.1));x.dispatchEvent(new Event('input',{bubbles:true}))}")
    page.wait_for_timeout(50); after=page.locator("[data-demo-output]").inner_text()
    result.update({"broken_svg":broken,"interaction_changed":before!=after,"console_errors":errors})
    page.screenshot(path=str(out/f"{slug}-1440.png"),full_page=True)
    if result["svg_count"]<8 or broken or before==after or errors: report["status"]="failed"
    report["articles"][slug]=result; page.close()
   browser.close()
 finally: server.shutdown(); server.server_close()
 (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if report["status"]=="passed" else 1
if __name__=="__main__": raise SystemExit(main())
