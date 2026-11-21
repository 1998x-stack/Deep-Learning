#!/usr/bin/env python3
"""Static integrity checks for HTML, MathML, SVG and evidence records."""
import json,xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SITE=ROOT/"site"
SLUGS=["xor","gradient","cost-functions","output-units","hidden-units","relu","sigmoid-tanh","other-activations"]
class Audit(HTMLParser):
 def __init__(self): super().__init__(); self.images=[]; self.math=0; self.h1=0; self.external=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=="img": self.images.append(a.get("src",""))
  if tag=="math": self.math+=1
  if tag=="h1": self.h1+=1
  for k in ("src","href"):
   v=a.get(k,"")
   if v.startswith(("https://cdn.","https://fonts.googleapis.com","http://")): self.external.append(v)
def main():
 errors=[]
 if not (SITE/"index.html").is_file(): errors.append("missing site/index.html")
 for slug in SLUGS:
  page=SITE/"articles"/slug/"index.html"
  if not page.is_file(): errors.append(f"missing article {slug}"); continue
  a=Audit(); a.feed(page.read_text(encoding="utf-8")); refs=[x for x in a.images if x.endswith(".svg")]
  if len(refs)<8: errors.append(f"{slug}: expected >=8 SVG references, got {len(refs)}")
  if a.math<8: errors.append(f"{slug}: expected >=8 MathML blocks, got {a.math}")
  if a.h1!=1: errors.append(f"{slug}: expected one h1, got {a.h1}")
  if a.external: errors.append(f"{slug}: external runtime dependency {a.external}")
  for src in refs:
   if not (page.parent/src).resolve().is_file(): errors.append(f"{slug}: missing image {src}")
  ev=SITE/"assets"/"data"/slug/"evidence.json"
  try:
   data=json.loads(ev.read_text(encoding="utf-8"))
   if len(data.get("claims",[]))<2: errors.append(f"{slug}: evidence ledger too short")
   for c in data["claims"]:
    if c.get("status") not in {"verified","derived","illustrative","unverified"} or not c.get("source_url") or not c.get("locator"): errors.append(f"{slug}: incomplete evidence claim")
  except Exception as e: errors.append(f"{slug}: invalid evidence JSON: {e}")
  svgdir=SITE/"assets"/"svg"/slug; svgs=list(svgdir.glob("*.svg"))
  if len(svgs)<8: errors.append(f"{slug}: fewer than eight standalone SVGs")
  for file in svgs:
   try:
    root=ET.parse(file).getroot(); ns={"s":"http://www.w3.org/2000/svg"}
    if root.tag!="{http://www.w3.org/2000/svg}svg" or not root.get("viewBox"): errors.append(f"{file}: invalid root or viewBox")
    if root.find("s:title",ns) is None or root.find("s:desc",ns) is None: errors.append(f"{file}: missing title/desc")
    ids=[n.get("id") for n in root.iter() if n.get("id")]
    if len(ids)!=len(set(ids)): errors.append(f"{file}: duplicate IDs")
   except Exception as e: errors.append(f"{file}: invalid XML {e}")
 if errors:
  print("SITE VALIDATION FAILED"); [print(" -",e) for e in errors]; return 1
 print(f"SITE VALIDATION PASSED: {len(SLUGS)} articles, 64 SVGs, MathML and evidence checks."); return 0
if __name__=="__main__": raise SystemExit(main())
