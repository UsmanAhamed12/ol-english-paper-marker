# ruff: noqa: E501
"""Loopback-only HTTP interface for private visual evidence labeling.

Line-length checking is disabled because this module embeds dependency-free HTML,
CSS, and JavaScript served only by the loopback annotation tool.
"""

from __future__ import annotations

import json
import mimetypes
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    EvidenceAnnotation,
    EvidenceAnnotationRepository,
    EvidenceAnnotationStore,
)
from app.evaluation.evidence_benchmark.models import EvidenceBenchmarkManifest
from app.evaluation.evidence_benchmark.reverification import (
    EvidenceReverificationRepository,
)
from app.evaluation.evidence_expansion.models import EvidenceExpansionManifest
from app.evaluation.teacher_evidence.models import TeacherEvidenceManifest

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_ANNOTATION_PORT = 8765
MAX_REQUEST_BYTES = 64 * 1024


class EvidenceAnnotationServer(ThreadingHTTPServer):
    """HTTP server carrying only private benchmark state."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        manifest: (
            EvidenceBenchmarkManifest
            | EvidenceExpansionManifest
            | TeacherEvidenceManifest
        ),
        private_root: Path,
        repository: EvidenceAnnotationRepository,
        reverification_repository: EvidenceReverificationRepository | None,
    ) -> None:
        super().__init__(address, handler)
        self.manifest = manifest
        self.private_root = private_root.resolve()
        self.repository = repository
        self.reverification_repository = reverification_repository


def create_annotation_server(
    manifest: EvidenceBenchmarkManifest
    | EvidenceExpansionManifest
    | TeacherEvidenceManifest,
    private_root: Path,
    repository: EvidenceAnnotationRepository,
    *,
    port: int = DEFAULT_ANNOTATION_PORT,
    reverification_repository: EvidenceReverificationRepository | None = None,
) -> EvidenceAnnotationServer:
    """Build a server bound exclusively to IPv4 loopback."""

    if not 0 <= port <= 65535:
        raise ValueError("Annotation port must be between 0 and 65535")
    return EvidenceAnnotationServer(
        (LOOPBACK_HOST, port),
        EvidenceAnnotationRequestHandler,
        manifest=manifest,
        private_root=private_root,
        repository=repository,
        reverification_repository=reverification_repository,
    )


class EvidenceAnnotationRequestHandler(BaseHTTPRequestHandler):
    """Serve the local UI, private images, and validated label writes."""

    server: EvidenceAnnotationServer

    def do_GET(self) -> None:  # noqa: N802
        if not self._valid_host():
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid local host"})
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._bytes(
                HTTPStatus.OK, ANNOTATION_HTML.encode(), "text/html; charset=utf-8"
            )
        elif path == "/api/benchmark":
            self._json(HTTPStatus.OK, _benchmark_payload(self.server))
        elif path.startswith("/assets/"):
            self._asset(path)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._valid_host():
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid local host"})
            return
        path = urlsplit(self.path).path
        prefix = "/api/annotations/"
        if not path.startswith(prefix):
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        sample_id = unquote(path[len(prefix) :])
        if sample_id not in {
            sample.sample_id for sample in self.server.manifest.samples
        }:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown sample"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Invalid annotation request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Annotation body must be an object")
            reverified = payload.pop("reverified", None)
            if (
                self.server.reverification_repository is not None
                and reverified is not True
            ):
                raise ValueError("Explicit re-verification is required")
            payload["sample_id"] = sample_id
            annotation = EvidenceAnnotation.model_validate(payload)
            store = self.server.repository.save(annotation)
            if self.server.reverification_repository is not None:
                self.server.reverification_repository.mark_reverified(annotation)
        except (
            EvidenceSeparationError,
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid annotation"})
            return
        progress = _progress_payload(self.server, store)
        self._json(
            HTTPStatus.OK,
            {"saved": True, "sample_id": sample_id, **progress},
        )

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so private sample paths never reach terminals."""

    def _asset(self, request_path: str) -> None:
        parts = request_path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "assets":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        kind, filename = parts[1:]
        if kind not in {"samples", "overlays"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        valid_names = {
            f"{sample.sample_id}.png" for sample in self.server.manifest.samples
        }
        if filename not in valid_names:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        path = (self.server.private_root / kind / filename).resolve()
        if not path.is_relative_to(self.server.private_root) or not path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._bytes(HTTPStatus.OK, path.read_bytes(), content_type)

    def _valid_host(self) -> bool:
        host = self.headers.get("Host", "").split(":", maxsplit=1)[0]
        return host in {LOOPBACK_HOST, "localhost"}

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._bytes(
            status,
            json.dumps(payload, separators=(",", ":")).encode(),
            "application/json; charset=utf-8",
        )

    def _bytes(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; connect-src 'self'; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)


def _benchmark_payload(server: EvidenceAnnotationServer) -> dict[str, Any]:
    store = server.repository.load()
    stored = {
        item.sample_id: item.model_dump(mode="json") for item in store.annotations
    }
    reverified_ids = (
        server.reverification_repository.current_ids(store)
        if server.reverification_repository is not None
        else frozenset()
    )
    samples = []
    for sample in server.manifest.samples:
        overlay = server.private_root / "overlays" / f"{sample.sample_id}.png"
        samples.append(
            {
                "sample_id": sample.sample_id,
                "paper_alias": sample.paper_alias,
                "page_number": sample.page_number,
                "test_number": sample.test_number,
                "categories": [category.value for category in sample.categories],
                "width": sample.region.width,
                "height": sample.region.height,
                "sample_url": f"/assets/samples/{sample.sample_id}.png",
                "overlay_url": (
                    f"/assets/overlays/{sample.sample_id}.png"
                    if overlay.is_file()
                    else None
                ),
                "annotation": stored.get(sample.sample_id),
                "reverified": sample.sample_id in reverified_ids,
            }
        )
    return {
        "samples": samples,
        **_progress_payload(server, store),
        "mode": (
            "reverification"
            if server.reverification_repository is not None
            else "annotation"
        ),
        "local_only": True,
    }


def _progress_payload(
    server: EvidenceAnnotationServer, store: EvidenceAnnotationStore
) -> dict[str, object]:
    total = len(server.manifest.samples)
    if server.reverification_repository is None:
        completed = len(store.annotations)
    else:
        completed = len(server.reverification_repository.current_ids(store))
    distribution = Counter(item.evidence_type.value for item in store.annotations)
    return {
        "completed": completed,
        "total": total,
        "ready": completed == total,
        "class_distribution": dict(sorted(distribution.items())),
    }


ANNOTATION_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Private Evidence Labeler</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#eef2f6}
*{box-sizing:border-box}body{margin:0}.top{position:sticky;top:0;z-index:5;background:#172033;color:white;padding:12px 20px;display:flex;justify-content:space-between;gap:16px}
main{max-width:1500px;margin:auto;padding:18px}.card{background:white;border:1px solid #d7dee8;border-radius:10px;padding:16px;margin-bottom:14px;box-shadow:0 2px 8px #17203312}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px}.chip{display:inline-block;background:#e8eef7;border-radius:99px;padding:4px 9px;margin:4px}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:16px}.canvasWrap{overflow:auto;border:1px solid #bac5d3;background:#fafafa;max-height:70vh}canvas{display:block;width:100%;height:auto;cursor:crosshair}.overlay{width:100%;max-height:70vh;object-fit:contain;border:1px solid #bac5d3}.choices label{display:block;padding:7px 0}.buttons{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}button{border:0;border-radius:6px;padding:9px 13px;background:#315d9e;color:white;cursor:pointer}button.secondary{background:#5e6876}button.danger{background:#a13c3c}button:disabled{opacity:.45;cursor:not-allowed}.rect{display:flex;justify-content:space-between;gap:8px;padding:6px;border-bottom:1px solid #e5e9ef}.error{color:#a00000}.ok{color:#146b35}.hint{font-size:.92rem;color:#536174}.empty{padding:8px;background:#fff8dd;border-radius:6px}@media(max-width:900px){.grid{grid-template-columns:1fr}.top{position:static}}
</style></head><body>
<div class="top"><strong id="toolTitle">Private Evidence Labeler - local only</strong><span id="progress">Loading…</span></div>
<main><section class="card"><div class="meta"><strong id="sampleId"></strong><span id="paper"></span><span id="page"></span><span id="test"></span></div><div id="categories"></div></section>
<div class="grid"><section class="card"><h2>Original crop - draw student-answer rectangles</h2><div class="canvasWrap"><canvas id="canvas"></canvas></div><p class="hint">Drag directly on the image. Coordinates are saved relative to the original crop pixels.</p></section>
<aside><section class="card"><h2>Evidence class</h2><p id="currentClass" class="hint"></p><p id="reverifiedState" class="hint"></p><div id="classes" class="choices"></div></section>
<section class="card"><h2>Answer regions</h2><label class="empty"><input id="verifiedEmpty" type="checkbox"> Explicitly verified empty - no student-answer region</label><div id="rectangles"></div><div class="buttons"><button id="clear" class="danger">Clear rectangles</button></div></section>
<section class="card"><h2>Existing prediction overlay</h2><img id="overlay" class="overlay" alt="Existing local prediction overlay"><p id="noOverlay" class="hint">No overlay available.</p></section></aside></div>
<section class="card"><div id="message" role="status"></div><div class="buttons"><button id="previous" class="secondary">Previous sample</button><button id="next" class="secondary">Next sample</button><button id="save">Save</button><button id="saveNext">Save and next</button></div></section></main>
<script>
'use strict';
const classValues=['printed','student_candidate','teacher_candidate','unknown'];
let data=null,index=0,rectangles=[],dirty=false,drawing=null,image=new Image();
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d');
function current(){return data.samples[index]}
function render(){const s=current(),reverify=data.mode==='reverification';$('toolTitle').textContent=reverify?'Private Evidence Re-verification - local only':'Private Evidence Labeler - local only';$('sampleId').textContent=s.sample_id;$('paper').textContent='Paper: '+s.paper_alias;$('page').textContent='Page: '+s.page_number;$('test').textContent=s.test_number===null?'Test: not detected':'Test: '+String(s.test_number).padStart(2,'0');$('categories').replaceChildren(...s.categories.map(x=>{const e=document.createElement('span');e.className='chip';e.textContent=x;return e}));
 $('classes').innerHTML=classValues.map(v=>`<label><input type="radio" name="evidence" value="${v}"> ${v.toUpperCase()}</label>`).join('');
 const a=s.annotation;rectangles=a?a.answer_regions.map(r=>({...r.bbox})):[];$('verifiedEmpty').checked=!!a&&a.answer_status==='verified_empty';$('currentClass').textContent=a?'Current saved class: '+a.evidence_type.toUpperCase():'No current saved class.';$('reverifiedState').textContent=reverify?(s.reverified?'Re-verified in this session: YES':'Re-verified in this session: NO'):'';if(a&&!reverify){const radio=document.querySelector(`input[name=evidence][value="${a.evidence_type}"]`);if(radio)radio.checked=true}
 $('save').textContent=reverify?'Re-verified / Save':'Save';$('saveNext').textContent=reverify?'Re-verified / Save and next':'Save and next';
 document.querySelectorAll('input[name=evidence]').forEach(el=>el.onchange=()=>{dirty=true});$('overlay').hidden=!s.overlay_url;$('noOverlay').hidden=!!s.overlay_url;if(s.overlay_url)$('overlay').src=s.overlay_url+'?v='+Date.now();
 image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw()};image.src=s.sample_url+'?v='+Date.now();dirty=false;updateRectList();updateProgress();$('message').textContent='';$('previous').disabled=index===0;$('next').disabled=index===data.samples.length-1}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);ctx.lineWidth=Math.max(2,canvas.width/700);ctx.strokeStyle='#00a7d8';ctx.fillStyle='#00a7d822';for(const r of rectangles){ctx.fillRect(r.x,r.y,r.width,r.height);ctx.strokeRect(r.x,r.y,r.width,r.height)}if(drawing){ctx.strokeStyle='#ff8c00';ctx.strokeRect(drawing.x,drawing.y,drawing.width,drawing.height)}}
function point(ev){const b=canvas.getBoundingClientRect();return{x:Math.round((ev.clientX-b.left)*canvas.width/b.width),y:Math.round((ev.clientY-b.top)*canvas.height/b.height)}}
canvas.onpointerdown=ev=>{if($('verifiedEmpty').checked)return;const p=point(ev);drawing={x:p.x,y:p.y,width:0,height:0};canvas.setPointerCapture(ev.pointerId)};
canvas.onpointermove=ev=>{if(!drawing)return;const p=point(ev),x=Math.min(drawing.x,p.x),y=Math.min(drawing.y,p.y),w=Math.abs(p.x-drawing.x),h=Math.abs(p.y-drawing.y);drawing={x,y,width:w,height:h};draw()};
canvas.onpointerup=()=>{if(!drawing)return;if(drawing.width>=3&&drawing.height>=3){rectangles.push(drawing);dirty=true}drawing=null;draw();updateRectList()};
function updateRectList(){$('rectangles').replaceChildren(...rectangles.map((r,i)=>{const row=document.createElement('div');row.className='rect';const text=document.createElement('span');text.textContent=`${i+1}: ${r.x}, ${r.y}, ${r.width}, ${r.height}`;const del=document.createElement('button');del.className='danger';del.textContent='Delete';del.onclick=()=>{rectangles.splice(i,1);dirty=true;draw();updateRectList()};row.append(text,del);return row}))}
$('clear').onclick=()=>{rectangles=[];$('verifiedEmpty').checked=false;dirty=true;draw();updateRectList()};$('verifiedEmpty').onchange=()=>{if($('verifiedEmpty').checked)rectangles=[];dirty=true;draw();updateRectList()};
function move(delta){if(dirty&&!confirm('Discard unsaved changes for this sample?'))return;index=Math.max(0,Math.min(data.samples.length-1,index+delta));render()}$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
async function save(andNext){const selected=document.querySelector('input[name=evidence]:checked');if(!selected)return message('Choose exactly one evidence class after visual inspection.',false);const empty=$('verifiedEmpty').checked;if(!empty&&rectangles.length===0)return message('Draw at least one rectangle or explicitly verify empty.',false);const payload={evidence_type:selected.value,answer_status:empty?'verified_empty':'annotated',answer_regions:rectangles.map(b=>({bbox:b})),human_verified:true,reverified:data.mode==='reverification'};
 const response=await fetch('/api/annotations/'+current().sample_id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!response.ok)return message('Save failed. Check the annotation and retry.',false);const saved={...payload};delete saved.reverified;current().annotation=saved;current().reverified=data.mode==='reverification';const result=await response.json();data.completed=result.completed;data.ready=result.ready;data.class_distribution=result.class_distribution;if(data.mode==='reverification')$('reverifiedState').textContent='Re-verified in this session: YES';dirty=false;message(data.mode==='reverification'?'Re-verified and saved locally.':'Saved locally.',true);updateProgress();if(andNext&&index<data.samples.length-1){index++;render()}}
$('save').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);function message(text,ok){$('message').textContent=text;$('message').className=ok?'ok':'error'}function updateProgress(){const label=data.mode==='reverification'?'Re-verified':'Completed',d=data.class_distribution||{},classes=`P:${d.printed||0} S:${d.student_candidate||0} T:${d.teacher_candidate||0} U:${d.unknown||0}`;$('progress').textContent=`${index+1}/${data.total} viewed - ${label}: ${data.completed}/${data.total} - ${classes}`}
fetch('/api/benchmark').then(r=>r.json()).then(v=>{data=v;render()}).catch(()=>message('Unable to load the private benchmark.',false));
</script></body></html>"""
