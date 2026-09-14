#!/usr/bin/env python3
"""打印一次 aiperf 跑的平均 usage_prompt_cache_read_tokens —— 判断前缀缓存有没有命中。"""
import json, os, sys
for root, _, files in os.walk(sys.argv[1]):
    for fn in files:
        if fn.endswith(".json") and "aiperf" in fn:
            try:
                d = json.load(open(os.path.join(root, fn)))
            except Exception:
                continue
            r = d.get("records") or d
            v = (r.get("usage_prompt_cache_read_tokens") or {}).get("avg")
            if v is not None:
                print(int(v)); sys.exit(0)
print(0)
