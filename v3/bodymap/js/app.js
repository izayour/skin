// ---------- the flow: two photos in, one place on the body out --------------
// close-up  : the lesion itself, the thing that gets measured. Never analysed
//             here -- it is carried along so a saved mark can be recognised.
// wide shot : the same lesion at arm's length, with enough of the body in frame
//             for the pose model to find joints. This is the only photo the
//             location is computed from.

import {BodyView, pointToLocation} from "./body3d.js";
import {loadPose, detectPose, BONES, VISIBLE, facing} from "./pose.js";
import {locate, limbRois} from "./locate.js?v=2";
import {describe, SEG} from "./anatomy.js";
import {matFromImage, associate, findCardInRois} from "./pair.js?v=14";

const $=s=>document.querySelector(s);

// OpenCV is fetched at first use rather than at page load. Both wasm runtimes
// compiling at once (MediaPipe ~15 MB, OpenCV ~10 MB) locked the renderer hard
// enough that the page never became interactive; loading the second one only
// when a photo needs it keeps the two apart in time.
let _cv=null;
function ensureCv(){
  if(_cv) return _cv;
  _cv=new Promise((res,rej)=>{
    if(window.cv&&window.cv.imread) return res(window.cv);
    const s=document.createElement("script");
    s.src="../vendor/opencv.js";   // shared with the scanner; the only file outside bodymap/
    s.onload=()=>{
      const done=()=>res(window.cv);
      if(window.cv&&window.cv.getBuildInformation) done();
      else window.cv["onRuntimeInitialized"]=done;
    };
    s.onerror=()=>rej(new Error("opencv.js failed to load"));
    document.head.appendChild(s);
  });
  return _cv;
}
const state={close:null, wide:null, pose:null, tap:null, loc:null,
             conf:0, marks:load(), selected:null, manual:false,
             closeTap:null, closeCard:null, pxPerCmWide:null, paired:null,
             rulerRoi:null, pointingAtRuler:false};

const view=new BodyView($("#stage"));
// Reachable from the console the way __segTrail is in the main app: the 3D side
// has no text output of its own, so without a handle on it a wrong-looking
// marker cannot be told apart from a wrong location.
window.__bodymap={state, view, locate, describe, ensureCv, associate, matFromImage};
view.onPick=loc=>{
  // A hand correction outranks the photo -- and says so, rather than leaving a
  // stale "94% confident" next to a mark the user just moved themselves.
  state.loc=loc; state.manual=true; state.conf=1;
  render();
};

// ---------- photo intake -----------------------------------------------------
$("#closeFile").addEventListener("change",async e=>{
  const f=e.target.files[0]; if(!f) return;
  state.close=await readImage(f);
  state.closeTap=null; state.closeCard=null; state.paired=null;
  $("#closeHint").textContent=f.name;
  drawClose();
  status("Tap the lesion in the close-up.");
});

// The close-up's own ruler: four corners for the map, and cm ticks that turn
// the map's local stretch into real pixels-per-centimetre in the wide shot.
//
// Needs the lesion first, because the box it searches is defined relative to
// it. Run on the whole frame instead, detectRuler pools every paper-like
// contour into ONE minAreaRect and returns a rectangle spanning the photo with
// corners outside the image -- which looks like a successful detection right up
// until you print the numbers.
async function findCloseCard(bbox){
  state.closeCard=null;
  if(!bbox) return;
  await ensureCv();
  const m=matFromImage(state.close);
  try{
    // belowLesionRoi, exactly as the live app scopes it -- the ruler is laid
    // below the lesion by the same person following the same habit.
    const raw=belowLesionRoi(m.cols,m.rows,bbox);
    const card=findCardInRois(m,[{id:"below lesion",
      x:Math.max(0,Math.round(raw.x)), y:Math.max(0,Math.round(raw.y)),
      w:Math.min(m.cols,Math.round(raw.w)), h:Math.min(m.rows,Math.round(raw.h))}]);
    if(!card) return;
    state.closeCard={corners:card.corners, cmTicks:card.cmTicks,
                     sharpness:card.sharpness, pxPerCm:null};
    if(card.cmTicks&&card.cmTicks.length>1&&typeof twoCmTicksNear==="function"){
      const px=twoCmTicksNear(card.cmTicks, state.closeTap[0]);
      if(px) state.closeCard.pxPerCm=px.pxPerCm;
    }
  }catch(err){ console.warn("close-up ruler detection failed:",err); }
  finally{ m.delete(); }
}

$("#wideFile").addEventListener("change",async e=>{
  const f=e.target.files[0]; if(!f) return;
  status("reading photo…");
  state.wide=await readImage(f);
  state.tap=null; state.loc=null; state.manual=false;
  drawWide();
  status("finding the body…");
  try{
    state.pose=await detectPose(state.wide);
  }catch(err){
    console.error(err); state.pose=null;
    status("pose model failed to run — "+err.message);
  }
  drawWide();
  if(!state.pose){
    status("No body found in this photo. Use a wider shot, or place the mark by tapping the model.");
    return;
  }
  const f2=facing(state.pose);
  status(`Body found — camera is looking at the ${f2.front?"front":"back"}${f2.sure?"":" (uncertain)"}.`);
  await tryPair();                      // does nothing until the close-up is tapped
});

// ---------- tapping the lesion in the CLOSE-UP -------------------------------
// The easy tap: here the lesion is large and unmistakable. The ruler carries
// this point across to the wide shot, where hitting it by hand would be a
// coin toss on a limb only a dozen pixels wide.
const closeCv=$("#close");
closeCv.addEventListener("click",async e=>{
  if(!state.close) return;
  const r=closeCv.getBoundingClientRect();
  const sx=state.close.naturalWidth/r.width, sy=state.close.naturalHeight/r.height;
  state.closeTap=[(e.clientX-r.left)*sx, (e.clientY-r.top)*sy];
  drawClose();
  status("Finding the lesion and the ruler…");
  const bbox=await refineCloseTap();
  await findCloseCard(bbox);
  drawClose();
  if(!state.closeCard)
    status("No ruler found below the lesion in the close-up — tap the lesion in the wide shot instead.");
  await tryPair();
});

// Snap the tap to the lesion's own centroid using the live app's segmenter, so
// the point that gets mapped is the lesion rather than wherever a thumb landed.
// A failure here is not fatal: the raw tap is already good enough to map.
async function refineCloseTap(){
  await ensureCv();
  if(!state.closeTap) return null;
  const im=state.close, m=matFromImage(im);
  const half=Math.max(24,Math.round(Math.min(m.cols,m.rows)*0.06));
  const x=Math.max(0,Math.round(state.closeTap[0]-half));
  const y=Math.max(0,Math.round(state.closeTap[1]-half));
  const box={x, y, w:Math.min(m.cols-x,half*2), h:Math.min(m.rows-y,half*2)};
  const bg=Math.max(31,Math.floor(Math.min(box.w,box.h)*0.9))|1;
  let bbox=[box.x,box.y,box.w,box.h];        // fall back to the tap's own box
  if(typeof segmentLesion!=="function"){ m.delete(); return bbox; }
  try{
    const res=segmentLesion(m,box,bg);
    if(res&&res.centroid){
      state.closeTap=res.centroid.slice(); state.snapped=true; bbox=res.bbox;
    } else state.snapped=false;
  }catch(err){ console.warn("segmentation skipped:",err); state.snapped=false; }
  finally{ m.delete(); }
  return bbox;
}

function drawClose(){
  const im=state.close; if(!im) return;
  const cv2=closeCv, ctx=cv2.getContext("2d");
  cv2.width=im.naturalWidth; cv2.height=im.naturalHeight;
  const maxW=cv2.parentElement.clientWidth||520;
  cv2.style.width=Math.round(im.naturalWidth*Math.min(1,maxW/im.naturalWidth))+"px";
  cv2.style.height="auto";
  ctx.drawImage(im,0,0);
  const k=Math.max(2,im.naturalWidth/450);
  if(state.closeCard) outlineCard(ctx,state.closeCard.corners,"#3ddc97",k);
  if(state.closeTap) crosshair(ctx,state.closeTap[0],state.closeTap[1],k,"#ff3b30");
}

// ---------- the association --------------------------------------------------
async function tryPair(){
  state.paired=null; state.pxPerCmWide=null;
  if(!state.close||!state.wide||!state.closeTap) return;
  await ensureCv();
  status("Matching the ruler across the two photos…");
  const cm=matFromImage(state.close), wm=matFromImage(state.wide);
  // Boxes to hunt the ruler in: the one the user pointed at if they had to,
  // otherwise one per visible limb straight off the pose we already ran.
  const rois=state.rulerRoi ? [state.rulerRoi]
           : (state.pose ? limbRois(state.pose, wm.cols, wm.rows) : []);
  let res;
  try{ res=associate(cm,wm,state.closeTap,state.closeCard,rois); }
  catch(err){ console.error(err); res={ok:false,reason:"matching failed: "+err.message}; }
  finally{ cm.delete(); wm.delete(); }

  if(!res.ok){
    status(res.reason+" — tap the lesion in the wide shot instead.");
    drawWide(); return;
  }
  state.paired=res;
  state.pxPerCmWide=res.pxPerCmWide;
  state.tap={x:res.point[0], y:res.point[1]};
  state.manual=false;
  placeFromTap();
  drawWide(); render();
  status(`Ruler matched${res.pxPerCmWide?` — ${res.pxPerCmWide.toFixed(1)} px/cm at the lesion`:""}.`
        +" Check the red cross sits on the lesion in the wide shot.");
}

// One route for both the mapped point and a hand tap, so a fallback tap gets
// the measured scale too when the ruler was found but the map was not trusted.
function placeFromTap(){
  if(!state.pose||!state.tap){ state.loc=null; return; }
  const opts=state.pxPerCmWide?{pxPerMetre:state.pxPerCmWide*100}:null;
  const res=locate(state.tap,state.pose,opts);
  if(res.ok){
    state.loc={segment:res.segment,t:res.t,radial:res.radial};
    state.conf=res.confidence; state.alts=res.alternatives;
    state.cmFromNear=res.cmFromNear; state.near=res.near;
    state.scaleMeasured=res.scaleMeasured;
  }else{ state.loc=null; state.conf=0; status(res.reason); }
}

function outlineCard(ctx,corners,colour,k){
  ctx.strokeStyle=colour; ctx.lineWidth=k*1.6;
  ctx.beginPath(); ctx.moveTo(corners[0][0],corners[0][1]);
  for(let i=1;i<4;i++) ctx.lineTo(corners[i][0],corners[i][1]);
  ctx.closePath(); ctx.stroke();
  // Mark corner 0 -- if the two photos disagree about which corner that is, the
  // map is twisted, and this dot is the only way to see it at a glance.
  ctx.fillStyle=colour;
  ctx.beginPath(); ctx.arc(corners[0][0],corners[0][1],k*3,0,7); ctx.fill();
}
function crosshair(ctx,x,y,k,colour){
  ctx.strokeStyle=colour; ctx.lineWidth=k*1.6;
  ctx.beginPath(); ctx.arc(x,y,k*6,0,7); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x-k*10,y); ctx.lineTo(x-k*3,y); ctx.moveTo(x+k*3,y); ctx.lineTo(x+k*10,y);
  ctx.moveTo(x,y-k*10); ctx.lineTo(x,y-k*3); ctx.moveTo(x,y+k*3); ctx.lineTo(x,y+k*10);
  ctx.stroke();
}

// ---------- tapping the lesion in the wide shot ------------------------------
const wideCv=$("#wide");
wideCv.addEventListener("click",e=>{
  if(!state.wide) return;
  const r=wideCv.getBoundingClientRect();
  const sx=state.wide.naturalWidth/r.width, sy=state.wide.naturalHeight/r.height;
  const px=(e.clientX-r.left)*sx, py=(e.clientY-r.top)*sy;

  // Two different jobs for the same canvas. Pointing at the RULER needs no
  // precision at all -- the box just has to contain a large white card -- which
  // is the whole reason this fallback is acceptable where tapping the lesion
  // itself at this distance would not be.
  if(state.pointingAtRuler){
    const s=Math.round(Math.min(state.wide.naturalWidth,state.wide.naturalHeight)*0.30);
    const x=Math.max(0,Math.round(px-s/2)), y=Math.max(0,Math.round(py-s/2));
    state.rulerRoi={id:"pointed at",
      x, y, w:Math.min(state.wide.naturalWidth-x,s), h:Math.min(state.wide.naturalHeight-y,s)};
    state.pointingAtRuler=false;
    $("#pointRuler").textContent="Point at the ruler";
    drawWide(); tryPair(); return;
  }

  state.tap={x:px, y:py};
  state.manual=false;
  state.paired=null;                    // a hand tap overrides the ruler's answer
  placeFromTap();
  drawWide(); render();
});

$("#pointRuler").addEventListener("click",()=>{
  state.pointingAtRuler=!state.pointingAtRuler;
  state.rulerRoi=null;
  $("#pointRuler").textContent=state.pointingAtRuler
    ? "…now tap the ruler in the photo" : "Point at the ruler";
  if(state.pointingAtRuler) status("Tap roughly on the ruler in the wide shot — anywhere on it will do.");
});

// ---------- drawing the wide shot with the detection on top ------------------
function drawWide(){
  const im=state.wide; if(!im) return;
  const cv=wideCv, ctx=cv.getContext("2d");
  const maxW=cv.parentElement.clientWidth||640;
  const s=Math.min(1,maxW/im.naturalWidth);
  cv.width=im.naturalWidth; cv.height=im.naturalHeight;
  cv.style.width=Math.round(im.naturalWidth*s)+"px";
  cv.style.height="auto";
  ctx.drawImage(im,0,0);

  const k=Math.max(2,im.naturalWidth/450);      // keep overlay legible at any size
  if(state.pose){
    const P=state.pose.points;
    ctx.lineWidth=k*1.4; ctx.strokeStyle="rgba(90,200,255,.85)";
    for(const [a,b] of BONES){
      if(P[a].v<VISIBLE||P[b].v<VISIBLE) continue;
      ctx.beginPath(); ctx.moveTo(P[a].x,P[a].y); ctx.lineTo(P[b].x,P[b].y); ctx.stroke();
    }
    ctx.fillStyle="rgba(90,200,255,.95)";
    for(const p of P){ if(p.v<VISIBLE) continue;
      ctx.beginPath(); ctx.arc(p.x,p.y,k*1.7,0,7); ctx.fill(); }
  }
  // The matched card, with its corner 0 dotted. Comparing that dot against the
  // one on the close-up is how a twisted correspondence shows itself before it
  // becomes a mark on the wrong side of somebody's arm.
  if(state.rulerRoi){
    ctx.strokeStyle="rgba(255,176,32,.9)"; ctx.lineWidth=k*1.4;
    ctx.strokeRect(state.rulerRoi.x,state.rulerRoi.y,state.rulerRoi.w,state.rulerRoi.h);
  }
  if(state.paired) outlineCard(ctx,state.paired.wideCorners,"#3ddc97",k);
  if(state.tap) crosshair(ctx,state.tap.x,state.tap.y,k,"#ff3b30");
}

// ---------- result panel -----------------------------------------------------
function render(){
  const has=!!state.loc;
  $("#result").hidden=!has;
  $("#saveBtn").disabled=!has;
  if(has){
    // Centimetres only when the ruler supplied the scale. Quoting a distance
    // derived from "assume this person is 1.75 m" would look like a measurement
    // and be nothing of the kind.
    const cm=(state.scaleMeasured&&state.cmFromNear!=null&&state.near)
      ? ` — ${state.cmFromNear} cm from the ${state.near}` : "";
    $("#where").textContent=describe(state.loc)+cm;
    const pct=Math.round(state.conf*100);
    const how=state.paired ? "placed from the ruler · " : "";
    $("#conf").textContent = state.manual ? "placed by hand"
      : how+`${pct}% confident` + (state.alts&&state.alts.length&&pct<75
          ? ` — could also be the ${state.alts[0]}` : "");
    $("#conf").className = "conf "+(state.manual?"ok":pct>=75?"ok":pct>=45?"mid":"low");
    view.focus(state.loc);
  }
  view.setMarkers(
    state.marks.concat(has?[{id:"__new", loc:state.loc}]:[]),
    has?"__new":state.selected);
  drawMarkList();
}

// ---------- saved marks ------------------------------------------------------
function load(){ try{ return JSON.parse(localStorage.getItem("bodymap.marks")||"[]"); }
                 catch{ return []; } }
function save(){ localStorage.setItem("bodymap.marks",JSON.stringify(state.marks)); }

$("#saveBtn").addEventListener("click",()=>{
  if(!state.loc) return;
  state.marks.push({
    id:String(Date.now()),
    loc:state.loc,
    label:describe(state.loc),
    confidence:state.manual?1:state.conf,
    manual:state.manual,
    viaRuler:!!state.paired,
    cmFromNear:state.scaleMeasured?state.cmFromNear:null, near:state.near,
    when:new Date().toISOString(),
    thumb:state.close?thumbOf(state.close,180):null,
  });
  save();
  state.loc=null; state.tap=null; state.conf=0; state.manual=false;
  drawWide(); render();
  status("Saved. The mark stays on the model; add another photo for the next one.");
});

function drawMarkList(){
  const box=$("#marks");
  box.innerHTML="";
  $("#marksEmpty").hidden=state.marks.length>0;
  for(const m of state.marks.slice().reverse()){
    const el=document.createElement("div");
    el.className="mark"+(m.id===state.selected?" sel":"");
    el.innerHTML=`
      ${m.thumb?`<img src="${m.thumb}" alt="">`:`<div class="noimg">no photo</div>`}
      <div class="mtext">
        <b>${m.label}</b>
        <span>${new Date(m.when).toLocaleDateString()} · ${m.manual?"placed by hand":Math.round(m.confidence*100)+"% confident"}</span>
      </div>
      <button class="del" title="remove">×</button>`;
    el.addEventListener("click",ev=>{
      if(ev.target.classList.contains("del")){
        state.marks=state.marks.filter(x=>x.id!==m.id); save(); render(); return;
      }
      state.selected=m.id; view.focus(m.loc); render();
    });
    box.appendChild(el);
  }
}

// ---------- odds and ends ----------------------------------------------------
for(const b of document.querySelectorAll("[data-view]"))
  b.addEventListener("click",()=>view.view(b.dataset.view));
$("#ghost").addEventListener("input",e=>view.setOpacity(+e.target.value));

function status(t){ $("#status").textContent=t; }

function readImage(file){
  return new Promise((res,rej)=>{
    const r=new FileReader();
    r.onload=()=>{ const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=r.result; };
    r.onerror=rej; r.readAsDataURL(file);
  });
}
function thumbOf(im,max){
  const s=Math.min(1,max/Math.max(im.naturalWidth,im.naturalHeight));
  const c=document.createElement("canvas");
  c.width=Math.round(im.naturalWidth*s); c.height=Math.round(im.naturalHeight*s);
  c.getContext("2d").drawImage(im,0,0,c.width,c.height);
  return c.toDataURL("image/jpeg",0.7);
}

addEventListener("resize",()=>{drawWide(); drawClose();});

// The wasm and the .task together are ~15 MB; starting the fetch at page load
// means the wait happens while the user is still picking their first photo
// rather than after they have already asked for an answer.
loadPose(msg=>status(msg))
  .then(()=>status("Ready. Add the close-up (ruler in shot) and the wide shot, then tap the lesion in the close-up."))
  .catch(e=>status("Pose model unavailable ("+e.message+"). You can still place marks by tapping the model."));

render();
