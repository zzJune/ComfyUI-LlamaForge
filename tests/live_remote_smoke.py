"""Opt-in live smoke test for the configured Suibian remote model."""
import sys
import importlib.util
import importlib
from pathlib import Path
import json
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parent.parent))
spec=importlib.util.spec_from_file_location("_lf_p4_live",root/"__init__.py",submodule_search_locations=[str(root)])
pkg=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=pkg
spec.loader.exec_module(pkg)
remote=importlib.import_module(spec.name+".remote")
import torch

def main():
    cfg=remote.make_config("suibian","sakura-i2i")
    client=remote.RemoteClient("suibian","sakura-i2i")
    catalog=client.json("/v1/models")
    print("Authorized model aliases:",[m["id"] for m in catalog["data"]],flush=True)
    def test(prompt,image=None):
        t=time.monotonic()
        answer,raw,_=remote.REMOTE_SERVER.generate(cfg,"You are a helpful assistant.",prompt,128,0,0.95,42,240,False,think=False,image=image,vram_snapshot=True)
        stats=json.loads(raw)
        print(json.dumps({"answer":answer[:200],"elapsed_s":round(time.monotonic()-t,2),"usage":stats["usage"],"timings":stats["timings"],"performance":stats["performance"],"request_id":stats["request_id"]},ensure_ascii=False),flush=True)
        return answer
    assert "ok" in test("Reply with only the word OK.").lower()
    img=torch.zeros((1,224,224,3),dtype=torch.float32)
    img[:,:,:,0]=1
    assert "red" in test("What is the dominant color? Reply with a single English color word.",img).lower()
    print("LIVE REMOTE TEXT AND VISION PASSED",flush=True)

if __name__=="__main__":
    main()
