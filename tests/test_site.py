"""The static validator is a required regression gate."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_site_integrity():
 r=subprocess.run([sys.executable,str(ROOT/"scripts/validate_site.py")],cwd=ROOT,capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
