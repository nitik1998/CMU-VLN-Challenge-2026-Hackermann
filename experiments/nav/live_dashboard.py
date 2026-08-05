#!/usr/bin/env python3
"""Zero-dependency live viewer for a story-explorer run.

The event log is the source of truth. The browser receives it over SSE and can
attach rectangle-and-text feedback to the exact image and agent call at fault.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot perception live trace</title>
<style>
:root{--bg:#0b0e13;--panel:#131821;--line:#27303d;--text:#e9eef5;--muted:#94a0b2;
--observer:#55d6be;--occlusion:#f2b84b;--investigator:#8ab4ff;--repair:#ff7b93;
--tool:#c6a0f6;--move:#8bd450;--bad:#ff5f57;--good:#4cd97b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}
header{height:52px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 16px;gap:15px;background:#0d1118;position:sticky;top:0;z-index:20}
header h1{font-size:16px;margin:0}.live{display:flex;gap:7px;align-items:center;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 10px var(--good)}
#question{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#cbd5e1;margin-left:auto;max-width:48vw}
.layout{display:grid;grid-template-columns:minmax(420px,1.25fr) minmax(340px,.9fr) minmax(360px,1fr);height:calc(100vh - 52px)}
.pane{min-width:0;overflow:auto;border-right:1px solid var(--line)}.pane:last-child{border-right:0}.section{padding:14px}
.section h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}
#imageStage{position:relative;background:#05070a;min-height:240px;display:flex;align-items:center;justify-content:center;overflow:hidden}
#scene{max-width:100%;max-height:52vh;display:block}#mark{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
.imageName{font:12px ui-monospace,monospace;color:var(--muted);padding:8px 12px;border-top:1px solid var(--line);overflow-wrap:anywhere}
.feedback{border-top:1px solid var(--line);padding:12px}.feedback textarea{width:100%;height:64px;background:#0b1017;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px;resize:vertical}
button{background:#2368c4;border:0;border-radius:6px;color:white;padding:7px 11px;cursor:pointer}button:disabled{opacity:.45}.hint{font-size:12px;color:var(--muted);margin:6px 0}.saved{color:var(--good);margin-left:8px}
#timeline{padding:10px}.card{border:1px solid var(--line);border-left:4px solid #667085;background:var(--panel);border-radius:7px;margin-bottom:9px;overflow:hidden;cursor:pointer}.card.active{outline:1px solid #fff8}.cardHead{padding:9px 10px;display:flex;align-items:center;gap:8px}.badge{font-size:10px;text-transform:uppercase;letter-spacing:.08em;padding:2px 6px;border-radius:10px;background:#27303d}.tag{color:var(--muted);font:11px ui-monospace,monospace}.elapsed{margin-left:auto;color:var(--muted);font-size:11px}.preview{padding:0 10px 9px;color:#c5ceda;white-space:pre-wrap;max-height:86px;overflow:hidden}.streaming .badge:after{content:' …';color:var(--good)}
.observer{border-left-color:var(--observer)}.occlusion{border-left-color:var(--occlusion)}.investigator{border-left-color:var(--investigator)}.repair{border-left-color:var(--repair)}.tool{border-left-color:var(--tool)}.movement{border-left-color:var(--move)}.failure{border-left-color:var(--bad)}
#detail{padding:14px}.detailMeta{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:10px}.tabs{display:flex;gap:4px;margin:8px 0}.tabs button{background:#202733}.tabs button.active{background:#376fb9}.content{white-space:pre-wrap;overflow-wrap:anywhere;background:#0a0e14;border:1px solid var(--line);border-radius:7px;padding:11px;min-height:120px;max-height:69vh;overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,monospace}
.empty{color:var(--muted);padding:20px;text-align:center}.filter{width:100%;background:#0b1017;color:var(--text);border:1px solid var(--line);padding:8px;border-radius:6px;margin-bottom:9px}
@media(max-width:1100px){.layout{grid-template-columns:1fr 1fr}.pane:last-child{grid-column:1/3;border-top:1px solid var(--line)} }
</style></head><body>
<header><h1>Active perception trace</h1><div class="live"><span class="dot"></span><span id="conn">connecting</span></div><div id="question"></div></header>
<main class="layout">
 <section class="pane"><div class="section"><h2>Exact image being analyzed</h2></div><div id="imageStage"><div class="empty">Waiting for a panorama…</div><img id="scene" hidden><canvas id="mark"></canvas></div><div id="imageName" class="imageName">No image</div>
 <div class="feedback"><h2>Point out the error</h2><div class="hint">Drag a rectangle on the image, select the faulty agent step, and describe what is wrong. Feedback is saved with normalized coordinates.</div><textarea id="note" placeholder="Example: This is a fourth pillow hidden behind the table; the observer missed it."></textarea><div><button id="saveFeedback" disabled>Save feedback</button><span id="saved" class="saved"></span></div></div></section>
 <section class="pane"><div class="section"><h2>Agents, handoffs, tools and motion</h2><input id="filter" class="filter" placeholder="Filter observer, gate, movement, text…"></div><div id="timeline"></div></section>
 <section class="pane"><div id="detail"><h2>Selected event</h2><div class="empty">Click any step to inspect its complete prompt and raw response.</div></div></section>
</main>
<script>
const state={events:new Map(),cards:new Map(),calls:new Map(),selected:null,last:0,image:null,rect:null,boxes:[],tab:'response'};
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function artifact(v){return v&&typeof v==='object'&&v.artifact?v.artifact:null}
function fileUrl(v){const p=artifact(v)||v;return p?'/files/'+String(p).split('/').map(encodeURIComponent).join('/'):''}
function role(label='',kind='') {const s=(label+' '+kind).toLowerCase();if(s.includes('target_audit'))return['Blind target audit','investigator'];if(s.includes('observer'))return['Observer','observer'];if(s.includes('occlusion'))return['Blind occlusion audit','occlusion'];if(s.includes('investigator'))return['Investigator','investigator'];if(s.includes('revision')||s.includes('consistency'))return['Correction / gate','repair'];if(s.includes('zoom')||s.includes('sam')||s.includes('tool'))return['Visual tool','tool'];if(s.includes('move')||s.includes('capture'))return['Robot / camera','movement'];if(s.includes('error')||s.includes('failure'))return['Failure','failure'];return[label||kind||'Event','']}
function setImage(v){const p=artifact(v)||v;if(!p||state.image===p)return;state.image=p;state.rect=null;const img=$('#scene');img.src=fileUrl(v);img.hidden=false;$('#imageStage .empty')?.remove();$('#imageName').textContent=p;img.onload=draw;$('#saveFeedback').disabled=false}
function preview(e){if(e.kind==='agent_start')return 'Prompt sent to model';if(e.kind==='agent_complete')return e.raw||'';if(e.kind==='gate_reject')return e.issue||'';if(e.kind==='decision')return JSON.stringify(e.decision||{},null,2);if(e.kind.startsWith('movement'))return e.semantic_goal||JSON.stringify(e,null,2);return e.message||e.stage||JSON.stringify(e,null,2)}
function eventTitle(e){if(e.kind.startsWith('agent_'))return role(e.label,e.kind)[0];return role('',e.kind)[0]}
function finalJson(raw=''){for(let i=raw.indexOf('{');i>=0;i=raw.indexOf('{',i+1)){try{return JSON.parse(raw.slice(i))}catch(_){}}return null}
function claimedBoxes(e){const d=finalJson(e.raw||e.stream||'');if(!d)return[];return(d.task_relevant_visible||[]).filter(x=>Array.isArray(x.bbox_norm)&&x.bbox_norm.length===4).map(x=>({box:x.bbox_norm,label:x.instance_id||'claimed object'}))}
function addEvent(e){if(!e||state.events.has(e.id))return;state.events.set(e.id,e);state.last=Math.max(state.last,e.id||0);
 if(e.kind==='agent_token'){const c=state.calls.get(e.call_id);if(c){c.stream=(c.stream||'')+(e.text||'');refreshCall(c)}return}
 if(e.kind==='agent_complete'||e.kind==='agent_error'){let c=state.calls.get(e.call_id);if(!c){c={...e,kind:'agent_start'};state.calls.set(e.call_id,c);makeCard(c)}Object.assign(c,e,{complete:e.kind==='agent_complete'});refreshCall(c);if(state.selected===c)renderDetail(c);return}
 if(e.kind==='agent_start'){state.calls.set(e.call_id,e);makeCard(e);const imgs=e.images||[];if(imgs.length)setImage(imgs[0]);return}
 makeCard(e);if(e.kind==='capture_complete'&&e.image)setImage(e.image);if(e.kind==='run_start'&&e.question)$('#question').textContent=e.question}
function makeCard(e){const [title,cls]=role(e.label,e.kind);const node=document.createElement('div');node.className='card '+cls+(e.kind==='agent_start'?' streaming':'');node.dataset.search=(title+' '+(e.label||'')+' '+preview(e)).toLowerCase();node.innerHTML=`<div class="cardHead"><span class="badge">${esc(title)}</span><span class="tag">${esc(e.tag||e.kind)}</span><span class="elapsed">#${e.id||''}</span></div><div class="preview">${esc(preview(e)).slice(0,1800)}</div>`;node.onclick=()=>select(e,node);e._node=node;$('#timeline').append(node);applyFilter();if(e.kind==='agent_start')select(e,node)}
function refreshCall(e){const n=e._node;if(!n)return;n.classList.toggle('streaming',!e.complete&&!e.error);const text=e.raw||e.stream||'';n.querySelector('.preview').textContent=text;n.querySelector('.elapsed').textContent=(e.secs!=null?e.secs+'s · ':'')+'call '+e.call_id;n.dataset.search=(eventTitle(e)+' '+(e.label||'')+' '+text).toLowerCase();applyFilter()}
function select(e,node=e._node){document.querySelectorAll('.card.active').forEach(n=>n.classList.remove('active'));node?.classList.add('active');state.selected=e;state.boxes=claimedBoxes(e);const imgs=e.images||[];if(imgs.length)setImage(imgs[0]);draw();renderDetail(e)}
function renderDetail(e){let prompt=e.prompt||'',response=e.raw||e.stream||'',data=JSON.stringify(Object.fromEntries(Object.entries(e).filter(([k])=>!k.startsWith('_')&&!['prompt','raw','stream'].includes(k))),null,2);let body=state.tab==='prompt'?prompt:state.tab==='response'?(response||preview(e)):data;$('#detail').innerHTML=`<h2>${esc(eventTitle(e))}</h2><div class="detailMeta"><span>${esc(e.label||e.kind)}</span><span>${esc(e.tag||'')}</span><span>${e.secs!=null?esc(e.secs)+' seconds':''}</span><span>${e.out_tokens!=null?esc(e.out_tokens)+' output tokens':''}</span></div><div class="tabs"><button data-tab="prompt">Prompt</button><button data-tab="response">Live / raw response</button><button data-tab="data">Event JSON</button></div><div class="content">${esc(body||'(empty)')}</div>`;document.querySelectorAll('.tabs button').forEach(b=>{b.classList.toggle('active',b.dataset.tab===state.tab);b.onclick=()=>{state.tab=b.dataset.tab;renderDetail(e)}});const content=$('.content');content.scrollTop=content.scrollHeight}
function applyFilter(){const q=$('#filter').value.toLowerCase();document.querySelectorAll('.card').forEach(n=>n.hidden=q&&!n.dataset.search.includes(q))}
$('#filter').addEventListener('input',applyFilter);
const canvas=$('#mark'),ctx=canvas.getContext('2d'),img=$('#scene');
function draw(){const box=$('#imageStage').getBoundingClientRect();canvas.width=Math.round(box.width*devicePixelRatio);canvas.height=Math.round(box.height*devicePixelRatio);canvas.style.width=box.width+'px';canvas.style.height=box.height+'px';ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,box.width,box.height);if(!img.hidden&&img.complete){const ib=img.getBoundingClientRect(),ox=ib.left-box.left,oy=ib.top-box.top;for(const item of state.boxes){const den=Math.max(...item.box.map(Math.abs))<=1?1:1000,[x0,y0,x1,y1]=item.box.map(v=>Number(v)/den);const x=ox+x0*ib.width,y=oy+y0*ib.height,w=(x1-x0)*ib.width,h=(y1-y0)*ib.height;ctx.strokeStyle='#00e5ff';ctx.fillStyle='#00e5ff22';ctx.lineWidth=3;ctx.fillRect(x,y,w,h);ctx.strokeRect(x,y,w,h);ctx.fillStyle='#001014cc';ctx.fillRect(x,Math.max(0,y-21),Math.max(84,ctx.measureText(item.label).width+12),21);ctx.fillStyle='#00e5ff';ctx.font='bold 12px system-ui';ctx.fillText(item.label,x+5,Math.max(14,y-6))}}if(state.rect){ctx.strokeStyle='#ff3b30';ctx.lineWidth=3;ctx.fillStyle='#ff3b3025';ctx.fillRect(state.rect.x0,state.rect.y0,state.rect.x1-state.rect.x0,state.rect.y1-state.rect.y0);ctx.strokeRect(state.rect.x0,state.rect.y0,state.rect.x1-state.rect.x0,state.rect.y1-state.rect.y0)}}
let start=null;canvas.onpointerdown=e=>{const b=canvas.getBoundingClientRect();start={x:e.clientX-b.left,y:e.clientY-b.top};state.rect={x0:start.x,y0:start.y,x1:start.x,y1:start.y};canvas.setPointerCapture(e.pointerId);draw()};canvas.onpointermove=e=>{if(!start)return;const b=canvas.getBoundingClientRect();state.rect={x0:start.x,y0:start.y,x1:e.clientX-b.left,y1:e.clientY-b.top};draw()};canvas.onpointerup=e=>{start=null;draw()};addEventListener('resize',draw);
$('#saveFeedback').onclick=async()=>{if(!state.image||!state.rect)return;const b=canvas.getBoundingClientRect(),ib=img.getBoundingClientRect(),r=state.rect,ox=ib.left-b.left,oy=ib.top-b.top;const box=[(Math.min(r.x0,r.x1)-ox)/ib.width,(Math.min(r.y0,r.y1)-oy)/ib.height,(Math.max(r.x0,r.x1)-ox)/ib.width,(Math.max(r.y0,r.y1)-oy)/ib.height].map(v=>Math.max(0,Math.min(1,v)));const payload={image:state.image,box,call_id:state.selected?.call_id||null,event_id:state.selected?.id||null,agent:state.selected?.label||state.selected?.kind||null,note:$('#note').value};const res=await fetch('/api/feedback',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});$('#saved').textContent=res.ok?'Saved to feedback.jsonl':'Save failed';if(res.ok)$('#note').value=''};
async function boot(){const data=await fetch('/api/events').then(r=>r.json());data.forEach(addEvent);const es=new EventSource('/api/stream?after='+state.last);es.onopen=()=>$('#conn').textContent='live';es.onerror=()=>$('#conn').textContent='reconnecting';es.onmessage=m=>{try{addEvent(JSON.parse(m.data))}catch(e){}}}
boot();
</script></body></html>"""


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, run_dir: Path):
        super().__init__(address, handler)
        self.run_dir = run_dir.resolve()
        self.events = self.run_dir / "live_events.jsonl"
        self.feedback = self.run_dir / "feedback.jsonl"


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, fmt, *args):
        return

    def _reply(self, body: bytes, content_type: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _all_events(self):
        out = []
        if self.server.events.exists():
            for line in self.server.events.read_text(errors="replace").splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not out:
            out = self._legacy_trace_events()
        return out

    def _legacy_trace_events(self):
        """Make completed pre-dashboard runs inspectable (without fake streaming)."""
        trace_path = self.server.run_dir / "model_trace.json"
        if not trace_path.exists():
            return []
        try:
            calls = json.loads(trace_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        question_path = self.server.run_dir / "question.txt"
        question = (question_path.read_text(errors="replace").strip()
                    if question_path.exists() else "Historical model trace")
        events = [{"id": 1, "time": 0, "kind": "run_start",
                   "question": question, "replay": True}]
        event_id = 2

        def portable_image(raw_path):
            if not raw_path:
                return None
            path = Path(raw_path)
            try:
                return {"artifact": str(
                    path.resolve().relative_to(self.server.run_dir))}
            except ValueError:
                return None

        def append_call(call):
            nonlocal event_id
            images = []
            for raw_path in call.get("images") or []:
                image = portable_image(raw_path)
                if image:
                    images.append(image)
            common = {
                "call_id": call.get("n"), "label": call.get("label"),
                "tag": call.get("tag"), "images": images,
            }
            events.append({"id": event_id, "time": 0, "kind": "agent_start",
                           **common, "prompt": call.get("prompt", ""),
                           "in_tokens": call.get("in_tokens"),
                           "replay": True})
            event_id += 1
            events.append({"id": event_id, "time": 0,
                           "kind": "agent_complete", **common,
                           "raw": call.get("raw", ""),
                           "out_tokens": call.get("out_tokens"),
                           "secs": call.get("secs"), "replay": True})
            event_id += 1

        state_path = self.server.run_dir / "state.json"
        try:
            records = json.loads(state_path.read_text()) if state_path.exists() else []
        except (OSError, json.JSONDecodeError):
            records = []
        consumed = set()
        for record in records:
            iteration = int(record.get("iteration", 0))
            image = portable_image(record.get("image"))
            events.append({"id": event_id, "time": 0,
                           "kind": "capture_complete", "iteration": iteration,
                           "pose": record.get("pose"), "image": image,
                           "message": "Historical panorama capture",
                           "replay": True})
            event_id += 1
            prefix = f"view_{iteration:02d}"
            for index, call in enumerate(calls):
                if index not in consumed and str(call.get("tag") or "").startswith(prefix):
                    append_call(call)
                    consumed.add(index)
            if record.get("decision") is not None:
                events.append({"id": event_id, "time": 0, "kind": "decision",
                               "iteration": iteration,
                               "decision": record.get("decision"),
                               "candidate_viewpoints": record.get("candidates") or [],
                               "replay": True})
                event_id += 1
            if record.get("movement") is not None:
                movement = record["movement"]
                events.append({"id": event_id, "time": 0,
                               "kind": "movement_complete",
                               "iteration": iteration,
                               "candidate": movement.get("candidate"),
                               "semantic_goal": movement.get("semantic_goal"),
                               "expected_observation": movement.get(
                                   "expected_observation"),
                               "arrived": movement.get("arrived"),
                               "replay": True})
                event_id += 1
        for index, call in enumerate(calls):
            if index not in consumed:
                append_call(call)
        return events

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._reply(HTML.encode(), "text/html; charset=utf-8")
        if parsed.path == "/api/events":
            return self._reply(json.dumps(self._all_events()).encode(),
                               "application/json")
        if parsed.path == "/api/stream":
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            heartbeat = 0.0
            try:
                while True:
                    for event in self._all_events():
                        if int(event.get("id", 0)) > after:
                            data = json.dumps(event, ensure_ascii=False)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            after = int(event.get("id", after))
                    if time.time() - heartbeat > 10:
                        self.wfile.write(b": keepalive\n\n")
                        heartbeat = time.time()
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                return
        if parsed.path.startswith("/files/"):
            relative = unquote(parsed.path[len("/files/"):])
            path = (self.server.run_dir / relative).resolve()
            try:
                path.relative_to(self.server.run_dir)
            except ValueError:
                return self._reply(b"forbidden", "text/plain", HTTPStatus.FORBIDDEN)
            if not path.is_file():
                return self._reply(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return self._reply(path.read_bytes(), kind)
        return self._reply(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/api/feedback":
            return self._reply(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length))
            value["time"] = time.time()
            with self.server.feedback.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            self._reply(json.dumps({"saved": True}).encode(), "application/json")
        except (ValueError, json.JSONDecodeError) as exc:
            self._reply(json.dumps({"error": str(exc)}).encode(),
                        "application/json", HTTPStatus.BAD_REQUEST)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    server = DashboardServer(("127.0.0.1", args.port), Handler, run_dir)
    print(f"Live trace: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
