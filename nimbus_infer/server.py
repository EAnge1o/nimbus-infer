"""Browser visualisation demo, built on the standard library only.

    python -m nimbus_infer serve --model checkpoints/distilgpt2
    # then open http://127.0.0.1:8000/

Endpoints:

* ``GET /``              -- the single-page UI
* ``GET /api/info``      -- model metadata
* ``GET /api/stream``    -- Server-Sent Events: token-by-token text + live metrics
* ``GET /api/attention`` -- attention matrix (averaged over heads) for a prompt

Nothing here needs Flask, FastAPI, or any JavaScript build step.
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from .engine import Engine
from .sampling import SamplingParams
from .tokenizer import IncrementalDecoder

_MAX_ATTENTION_TOKENS = 16


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nimbus-infer demo</title>
<style>
:root{--bg:#0b1020;--panel:#141a2e;--line:#25304d;--fg:#e6ecff;--muted:#8ea0c9;--accent:#6ea8fe;--accent2:#f7b267}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--fg);
background:radial-gradient(1200px 600px at 20% -10%,#1b2547,#0b1020);min-height:100vh}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0;letter-spacing:.5px}
.badge{color:var(--muted);font-size:12px}
.wrap{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px 24px;max-width:1200px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
label{display:block;color:var(--muted);font-size:12px;margin:10px 0 4px}
textarea,input[type=number]{width:100%;background:#0d1428;color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:8px;font:inherit}
input[type=range]{width:100%;accent-color:var(--accent)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:var(--accent);color:#06122b;border:0;border-radius:8px;padding:9px 14px;font:inherit;font-weight:700;cursor:pointer}
button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
button:disabled{opacity:.45;cursor:default}
#output{white-space:pre-wrap;word-break:break-word;min-height:240px;font-size:15px;line-height:1.7}
#prompt_echo{color:var(--muted)}
.cursor{display:inline-block;width:8px;height:16px;background:var(--accent2);vertical-align:-2px;animation:blink .8s steps(2) infinite}
@keyframes blink{50%{opacity:0}}
.stats{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:12px}
.stats b{color:var(--fg)}
canvas{background:#0d1428;border:1px solid var(--line);border-radius:8px;max-width:100%}
.legend{color:var(--muted);font-size:12px}
.notice{display:none;margin:14px 24px 0 auto;max-width:1200px;padding:10px 14px;border:1px solid #6b4a1f;
background:#2a2113;color:var(--accent2);border-radius:8px;font-size:13px;line-height:1.6}
</style>
</head>
<body>
<header>
  <h1>nimbus-infer</h1>
  <span class="badge" id="modelinfo">loading…</span>
</header>
<div id="notice" class="notice"></div>
<div class="wrap">
  <div class="card">
    <label>Prompt</label>
    <textarea id="prompt" rows="4">The history of computing began</textarea>
    <label>temperature <span id="tempv">0.80</span></label>
    <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.8">
    <label>top-k <span id="topkv">40</span></label>
    <input type="range" id="top_k" min="0" max="100" step="1" value="40">
    <label>top-p <span id="toppv">0.95</span></label>
    <input type="range" id="top_p" min="0" max="1" step="0.01" value="0.95">
    <label>repetition penalty <span id="repv">1.10</span></label>
    <input type="range" id="repetition_penalty" min="1" max="2" step="0.05" value="1.1">
    <label>max new tokens</label>
    <input type="number" id="max_new_tokens" value="64" min="1" max="512">
    <div class="row" style="margin-top:14px">
      <button id="go">Generate</button>
      <button id="stop" class="ghost" disabled>Stop</button>
    </div>
  </div>
  <div class="card">
    <div id="output"><span id="prompt_echo"></span><span id="gen"></span><span class="cursor" id="cursor" style="display:none"></span></div>
    <div class="stats">
      <span>tokens <b id="s_tokens">0</b></span>
      <span>speed <b id="s_tps">0</b> tok/s</span>
      <span>time <b id="s_time">0.00</b> s</span>
      <span>kv cache <b id="s_kv">0.00</b> MB</span>
    </div>
  </div>
</div>
<div class="wrap" style="grid-template-columns:1fr">
  <div class="card">
    <div class="row">
      <button id="attn" class="ghost">Show attention (last layer)</button>
      <span class="legend">token × token attention, averaged over heads</span>
    </div>
    <canvas id="heat" width="640" height="470" style="margin-top:12px"></canvas>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const ranges=[["temperature","tempv",2],["top_k","topkv",0],["top_p","toppv",2],["repetition_penalty","repv",2]];
ranges.forEach(function(s){var el=$(s[0]);var up=function(){$ (s[1]).textContent=(+el.value).toFixed(s[2]);};el.oninput=up;up();});
// Scrolling the page must not silently change a slider value.
document.querySelectorAll("input[type=range]").forEach(function(el){
  el.addEventListener("wheel",function(e){e.preventDefault();},{passive:false});});
function setStat(tokens,tps,time,kv){$("s_tokens").textContent=tokens;$("s_tps").textContent=(tps||0).toFixed(1);
  $("s_time").textContent=(time||0).toFixed(2);$("s_kv").textContent=(kv||0).toFixed(2);}
fetch("/api/info").then(r=>r.json()).then(d=>{$("modelinfo").textContent=
  d.name+" · "+d.parameters.toLocaleString()+" params · "+d.architecture.layers+"L "+
  d.architecture.heads+"H (kv "+d.architecture.kv_heads+") · "+d.architecture.norm+"/"+
  d.architecture.activation+"/"+d.architecture.pos_encoding;
  if(d.random_weights){var n=$("notice");n.style.display="block";
    n.textContent="\u26a0 random weights: the text below is gibberish BY DESIGN \u2014 the engine works, the weights were never trained. "+
    "Run `nimbus_infer convert --repo distilgpt2` for coherent English.";}});

var es=null;
$("go").onclick=function(){
  var prompt=$("prompt").value;if(!prompt.trim())return;
  $("gen").textContent="";$("prompt_echo").textContent=prompt;
  $("cursor").style.display="inline-block";$("go").disabled=true;$("stop").disabled=false;
  var p=new URLSearchParams({prompt:prompt,temperature:$("temperature").value,top_k:$("top_k").value,
    top_p:$("top_p").value,repetition_penalty:$("repetition_penalty").value,
    max_new_tokens:$("max_new_tokens").value});
  es=new EventSource("/api/stream?"+p.toString());
  es.onmessage=function(e){var d=JSON.parse(e.data);
    if(d.type==="token"){$("gen").textContent+=d.text;setStat(d.index,d.tps,d.elapsed,0);}
    else if(d.type==="done"){setStat(d.tokens,d.tps,d.seconds,d.kv_cache_mb);stopStream();}};
  es.onerror=function(){stopStream();};
};
$("stop").onclick=stopStream;
function stopStream(){if(es){es.close();es=null;}$("cursor").style.display="none";
  $("go").disabled=false;$("stop").disabled=true;}

$("attn").onclick=function(){
  var prompt=$("prompt").value;if(!prompt.trim())return;
  fetch("/api/attention?"+new URLSearchParams({prompt:prompt})).then(r=>r.json()).then(drawHeat);
};
function drawHeat(d){
  var c=$("heat"),ctx=c.getContext("2d"),m=d.matrix,n=m.length;
  var left=72,top=64,pad=8;
  var cell=Math.max(6,Math.min((c.width-left-pad)/n,(c.height-top-pad)/n));
  ctx.clearRect(0,0,c.width,c.height);ctx.font="11px ui-monospace,monospace";
  var max=0;m.forEach(function(r){r.forEach(function(v){if(v>max)max=v;});});
  for(var i=0;i<n;i++)for(var j=0;j<n;j++){var v=m[i][j]/(max||1);
    ctx.fillStyle="rgb("+Math.round(20+v*235)+","+Math.round(30+v*140)+","+Math.round(70+v*120)+")";
    ctx.fillRect(left+j*cell,top+i*cell,cell-1,cell-1);}
  ctx.fillStyle="#8ea0c9";
  if(cell>=16){ctx.textAlign="right";
    for(var i2=0;i2<n;i2++)ctx.fillText(d.tokens[i2],left-8,top+i2*cell+cell/2+4);
    for(var j2=0;j2<n;j2++){ctx.save();ctx.translate(left+j2*cell+cell/2,top-8);ctx.rotate(-Math.PI/4);
      ctx.textAlign="left";ctx.fillText(d.tokens[j2],0,0);ctx.restore();}}
  ctx.textAlign="left";ctx.fillText("query →",left,top-46);ctx.fillText("key ↓",10,top-12);
}
</script>
</body>
</html>
"""


def _sampling_params(query: dict) -> SamplingParams:
    def get(name: str, cast, default):
        raw = query.get(name, [None])[0]
        if raw in (None, ""):
            return default
        return cast(raw)

    return SamplingParams(
        max_new_tokens=get("max_new_tokens", int, 48),
        temperature=get("temperature", float, 0.8),
        top_k=get("top_k", int, 40),
        top_p=get("top_p", float, 0.95),
        repetition_penalty=get("repetition_penalty", float, 1.1),
        seed=get("seed", int, None),
    )


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "nimbus-infer"

    def log_message(self, *args) -> None:  # keep the console tidy
        pass

    @property
    def engine(self) -> Engine:
        return self.server.engine  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            return self._send_html(PAGE)
        if parsed.path == "/api/info":
            return self._send_json(self.engine.info())
        if parsed.path == "/api/attention":
            return self._attention(query)
        if parsed.path == "/api/stream":
            return self._stream(query)
        return self._send_json({"error": "not found"}, status=404)

    # -- responses -------------------------------------------------------
    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- endpoints -------------------------------------------------------
    def _stream(self, query: dict) -> None:
        prompt = query.get("prompt", [""])[0]
        if not prompt.strip():
            return self._send_json({"error": "empty prompt"}, status=400)

        params = _sampling_params(query)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True  # the stream is one-shot; close when done

        tokenizer = self.engine.tokenizer
        start = time.perf_counter()
        prompt_tokens = len(self.engine.encode(prompt))
        decoder = IncrementalDecoder(tokenizer) if tokenizer else None
        produced = 0
        try:
            self._sse({"type": "start", "prompt_tokens": prompt_tokens})
            for token in self.engine.iter_token_ids(prompt, params):
                produced += 1
                text = decoder.push(token) if decoder else f"{token} "
                elapsed = time.perf_counter() - start
                self._sse({
                    "type": "token",
                    "text": text,
                    "index": produced,
                    "tps": produced / elapsed if elapsed > 0 else 0.0,
                    "elapsed": elapsed,
                })
            if decoder:
                tail = decoder.flush()
                if tail:
                    self._sse({"type": "token", "text": tail, "index": produced, "tps": 0.0,
                               "elapsed": time.perf_counter() - start})
            elapsed = time.perf_counter() - start
            self._sse({
                "type": "done",
                "tokens": produced,
                "seconds": elapsed,
                "tps": produced / elapsed if elapsed > 0 else 0.0,
                "kv_cache_mb": self.engine.model.new_cache().bytes / (1024 ** 2),
            })
        except (BrokenPipeError, ConnectionResetError):  # client hit Stop
            return

    def _attention(self, query: dict) -> None:
        prompt = query.get("prompt", [""])[0]
        if not prompt.strip():
            return self._send_json({"error": "empty prompt"}, status=400)
        layer = int(query.get("layer", ["-1"])[0])

        ids = self.engine.encode(prompt)[:_MAX_ATTENTION_TOKENS]
        self.engine.model.forward(ids, capture_attention=True)
        captured = self.engine.model.captured_attention
        matrix = captured[layer]
        tokens = (
            self.engine.tokenizer.token_strings(ids)
            if self.engine.tokenizer
            else [str(i) for i in ids]
        )
        return self._send_json({
            "tokens": tokens,
            "matrix": np.round(matrix, 4).tolist(),
            "n_layers": len(captured),
        })


def create_server(engine: Engine, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), DemoHandler)
    httpd.engine = engine  # type: ignore[attr-defined]
    return httpd


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    httpd = create_server(engine, host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"nimbus-infer demo running at {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        httpd.server_close()