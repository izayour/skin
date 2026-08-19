// ---------- from a tap on a photograph to a place on the body ---------------
//
// Input: the pixel the lesion sits on in the wide shot, plus the pose landmarks
// for that same shot. Output: {segment, t, radial} in the canonical body of
// anatomy.js -- which limb, how far along it, and which way round it.
//
// The whole method rests on one observation: a photograph of a limb is, near
// enough, a photograph of a cylinder. Two landmarks give the cylinder's axis in
// the image, so the tap resolves into a distance ALONG that axis (which is t,
// and survives projection well) and a distance ACROSS it (which is a chord of
// the circular cross section, and therefore fixes the angle round the limb up
// to the near/far ambiguity that no single photo can resolve).
//
// That ambiguity is settled by which way the subject faces -- the camera sees
// the front of everything or the back of everything -- and that is a fact
// MediaPipe hands us, not one this code guesses per limb.
//
// What this genuinely cannot do: recover an angle when the limb points at the
// camera (the axis collapses to a point in the image), or place a mark on a
// body part the pose model never found. Both come back as low confidence rather
// than as a confident wrong answer, because a wrong answer here is worse than
// no answer -- it sends someone to look at the wrong arm.

import {SEGMENTS, SEG, JOINTS, LM, torsoProfileAt,
        sub, add, mul, dot, len, norm, lerp3, radiusAt, segLength} from "./anatomy.js";
import {VISIBLE, facing} from "./pose.js";

// ---------- 2D helpers -------------------------------------------------------
const v2  = (a,b)=>[b.x-a.x, b.y-a.y];
const l2  = a=>Math.hypot(a[0],a[1]);
const n2  = a=>{const l=l2(a)||1;return [a[0]/l,a[1]/l];};
const d2  = (a,b)=>a[0]*b[0]+a[1]*b[1];

// How many image pixels one metre of body spans. Taken as the median over every
// segment both of whose landmarks are visible: any single segment can be
// foreshortened to nothing by pointing at the camera, and using such a segment's
// own length as the scale would shrink its apparent width to match, making a tap
// anywhere near it look like a perfect hit.
function pixelsPerMetre(pose){
  const s=[];
  for(const seg of SEGMENTS){
    if(seg.kind!=="limb"||!seg.lm) continue;
    const [i,j]=seg.lm;
    if(pose.points[i].v<VISIBLE||pose.points[j].v<VISIBLE) continue;
    const px=l2(v2(pose.points[i],pose.points[j])), m=segLength(seg);
    if(px>4&&m>0.01) s.push(px/m);
  }
  if(!s.length){
    // No usable limb. Shoulder span is the last resort; it is nearly always
    // the widest thing visible and it is not foreshortened in a frontal shot.
    const P=pose.points;
    if(P[LM.shoulder_l].v>=VISIBLE&&P[LM.shoulder_r].v>=VISIBLE)
      return l2(v2(P[LM.shoulder_l],P[LM.shoulder_r]))/0.37;
    return null;
  }
  s.sort((a,b)=>a-b);
  return s[s.length>>1];
}

// ---------- where to go looking for the ruler --------------------------------
// One tight box per visible limb, plus the torso. The card is lying ON the
// body, so one of these contains it -- and each is the kind of view detectRuler
// was built for: mostly skin, one card, nothing else. Handed the whole frame it
// pools every paper-like contour in the room into a single rectangle instead.
//
// Padded by a couple of limb-widths because the card sits BESIDE the lesion,
// overhanging the limb rather than tucked inside its silhouette.
export function limbRois(pose, w, h){
  const P=pose.points, ppm=pixelsPerMetre(pose), out=[];
  const clamp=(x,y,bw,bh)=>{
    const x0=Math.max(0,Math.round(x)), y0=Math.max(0,Math.round(y));
    const x1=Math.min(w,Math.round(x+bw)), y1=Math.min(h,Math.round(y+bh));
    return (x1-x0>=48&&y1-y0>=48) ? {x:x0,y:y0,w:x1-x0,h:y1-y0} : null;
  };
  for(const seg of SEGMENTS){
    if(seg.kind!=="limb"||!seg.lm) continue;
    const [i,j]=seg.lm;
    if(P[i].v<VISIBLE||P[j].v<VISIBLE) continue;
    const pad=Math.max(30, (ppm?radiusAt(seg,0.5)*ppm:30)*2.6);
    const r=clamp(Math.min(P[i].x,P[j].x)-pad, Math.min(P[i].y,P[j].y)-pad,
                  Math.abs(P[i].x-P[j].x)+2*pad, Math.abs(P[i].y-P[j].y)+2*pad);
    if(r) out.push(Object.assign({id:seg.id},r));
  }
  const sL=P[LM.shoulder_l], sR=P[LM.shoulder_r], hL=P[LM.hip_l], hR=P[LM.hip_r];
  if(sL.v>=VISIBLE&&sR.v>=VISIBLE&&hL.v>=VISIBLE&&hR.v>=VISIBLE){
    const xs=[sL.x,sR.x,hL.x,hR.x], ys=[sL.y,sR.y,hL.y,hR.y];
    const pad=Math.max(30,(ppm?0.05*ppm:30));
    const r=clamp(Math.min(...xs)-pad, Math.min(...ys)-pad,
                  Math.max(...xs)-Math.min(...xs)+2*pad,
                  Math.max(...ys)-Math.min(...ys)+2*pad);
    if(r) out.push(Object.assign({id:"torso"},r));
  }
  // Largest last: a tight box is worth more than a loose one, and the search
  // stops at the first convincing card.
  out.sort((a,b)=>a.w*a.h-b.w*b.h);
  return out;
}

// ---------- the match --------------------------------------------------------
// opts.pxPerMetre, when the ruler has been found in this photo, is a MEASURED
// scale rather than the guess below -- and it is the single number the whole
// match is most sensitive to, since it sets every limb's width, which decides
// both which limb wins and the angle round it.
export function locate(tap, pose, opts){
  const P=pose.points;
  const ppm = (opts&&opts.pxPerMetre) || pixelsPerMetre(pose);
  if(!ppm) return {ok:false, reason:"no usable landmarks in this photo"};
  const face = facing(pose);

  const cands=[];

  // --- limbs: project onto the axis, measure the offset across it
  for(const seg of SEGMENTS){
    if(seg.kind!=="limb"||!seg.lm) continue;
    const [i,j]=seg.lm;
    if(P[i].v<VISIBLE||P[j].v<VISIBLE) continue;
    const A=P[i], B=P[j], ab=v2(A,B), L=l2(ab);
    if(L<8) continue;                       // pointing at the camera: no axis
    const u=n2(ab), ap=[tap.x-A.x, tap.y-A.y];
    const tRaw = d2(ap,u)/L;
    const t = Math.max(0,Math.min(1,tRaw));
    const off = ap[0]*(-u[1]) + ap[1]*(u[0]);   // signed, +90deg from the axis
    const halfW = radiusAt(seg,t)*ppm;
    if(halfW<1) continue;

    // Off the end of a limb is a miss; off the side by more than its radius is
    // a miss. Both are charged in units of the limb's own width, so a tap 2 cm
    // off a finger is not judged as leniently as 2 cm off a thigh.
    //
    // The overshoot is measured in PIXELS and only then divided by the width.
    // Charging it in fractions of the segment's own on-screen length is what a
    // foreshortened part exploits: a hand pointing at the camera is 17 px long,
    // so running three hand-lengths past its fingertips cost less than a
    // correct hit on the forearm, and taps near the wrist came back as "hand".
    const across = Math.abs(off)/halfW;
    const overPx = (tRaw<0 ? -tRaw : tRaw>1 ? tRaw-1 : 0)*L;
    const cost   = across + (overPx/halfW)*0.6;
    cands.push({seg, t, off, halfW, cost, axis2:u, L});
  }

  // --- torso: a quad, not a cylinder
  const tor = torsoCandidate(tap, pose, ppm);
  if(tor) cands.push(tor);

  // --- head: a disc around the nose, sized off the ear span when we have it
  const head = headCandidate(tap, pose, ppm);
  if(head) cands.push(head);

  if(!cands.length) return {ok:false, reason:"no body part visible near that point"};
  cands.sort((a,b)=>a.cost-b.cost);
  const best=cands[0];

  // Confidence is honest about two separate doubts: how far outside the part
  // the tap fell, and how much better the runner-up was NOT. A tap in the gap
  // between two touching limbs scores well on both counts yet is a coin flip,
  // and that has to show in the number.
  const margin = cands.length>1 ? cands[1].cost-best.cost : 1.5;
  let conf = Math.max(0, Math.min(1, (1.35-best.cost)/1.35))
           * Math.max(0.35, Math.min(1, 0.45+margin));
  if(!face.sure) conf*=0.75;

  const loc = best.kind==="torso" ? torsoLoc(best,face)
            : best.kind==="head"  ? headLoc(best,face)
            :                       limbLoc(best,face);

  // Centimetres from the near joint, measured in the photo. Worth far more than
  // "62% along" at a follow-up: a percentage moves when the pose does, a
  // distance from the elbow crease does not. Only honest when the scale was
  // measured off the ruler, and still a lower bound -- a limb angled toward the
  // camera is shorter on screen than it is in life.
  const cmFromNear = (best.seg && best.L)
    ? +(best.t*best.L*100/ppm).toFixed(1) : null;

  return {ok:true, ...loc, confidence:conf, facing:face,
          cmFromNear, near:best.seg?best.seg.near:null,
          scaleMeasured:!!(opts&&opts.pxPerMetre),
          alternatives:cands.slice(1,3).map(c=>SEG[(c.seg||{}).id||c.id]?.label||c.id)
                             .filter(Boolean),
          debug:{cost:best.cost, margin, ppm}};
}

// The step that actually leaves the plane. `off` is a chord across a circle of
// radius halfW, so sqrt(halfW^2 - off^2) is how far the surface stands toward
// the camera at that chord -- giving a 3D direction without ever needing depth
// from the photo. Screen axes map to world axes differently front and back,
// which is why `facing` is threaded all the way down here.
function radialFromOffset(seg, t, off, halfW, front){
  const A=JOINTS[seg.a], B=JOINTS[seg.b];
  const axis=norm(sub(B,A));
  const r=Math.max(-1,Math.min(1, off/halfW));
  const toward = Math.sqrt(Math.max(0,1-r*r));

  // Image +x is the subject's left (+X) seen from the front, their right when
  // seen from behind; image +y is always downward, i.e. world -Y.
  const sx = front? 1 : -1;
  const zc = front? 1 : -1;

  // The in-image perpendicular, lifted into world space and then made truly
  // perpendicular to the 3D axis (the photo's perpendicular is only
  // perpendicular in projection).
  const u=perp2OfSegment(seg);
  let side=[sx*u[0]*r, -u[1]*r, 0];
  side=sub(side, mul(axis, dot(side,axis)));
  const out=norm(add(side, [0,0,zc*toward]));
  return out;
}
// The direction across the limb, expressed in the canonical model rather than
// in the photo: the model's own limbs lie in the XY plane, so their in-image
// perpendicular is known exactly and does not have to be recovered from pixels.
function perp2OfSegment(seg){
  const A=JOINTS[seg.a], B=JOINTS[seg.b];
  const ax=[B[0]-A[0], B[1]-A[1]];
  const l=Math.hypot(ax[0],ax[1])||1;
  return [-(ax[1]/l), (ax[0]/l)];
}

function limbLoc(c, face){
  const radial = radialFromOffset(c.seg, c.t, c.off, c.halfW, face.front);
  return {segment:c.seg.id, t:c.t, radial};
}

// ---------- torso ------------------------------------------------------------
// Shoulders and hips give a quad. Rather than invert a bilinear map, walk down
// the shoulder-centre-to-hip-centre axis for v and measure across for u: the
// torso is close enough to a ruled surface for that, and it degrades gracefully
// when one hip is hidden.
function torsoCandidate(tap, pose, ppm){
  const P=pose.points;
  const sL=P[LM.shoulder_l], sR=P[LM.shoulder_r], hL=P[LM.hip_l], hR=P[LM.hip_r];
  if(sL.v<VISIBLE||sR.v<VISIBLE) return null;
  const sC={x:(sL.x+sR.x)/2, y:(sL.y+sR.y)/2};
  const haveHips = hL.v>=VISIBLE&&hR.v>=VISIBLE;
  const hC = haveHips ? {x:(hL.x+hR.x)/2, y:(hL.y+hR.y)/2}
                      : {x:sC.x, y:sC.y+0.45*ppm};   // fall back to model length
  const ax=v2(sC,hC), L=l2(ax);
  if(L<8) return null;
  const u=n2(ax), ap=[tap.x-sC.x, tap.y-sC.y];
  const vRaw=d2(ap,u)/L;
  const v=Math.max(0,Math.min(1,vRaw));
  const off= ap[0]*(-u[1]) + ap[1]*(u[0]);
  const {w}=torsoProfileAt(v);
  const halfW=w*ppm;
  const overPx=(vRaw<0?-vRaw:vRaw>1?vRaw-1:0)*L;   // pixels, as for the limbs
  return {kind:"torso", id:"torso", t:v, off, halfW,
          cost:Math.abs(off)/halfW + (overPx/halfW)*0.6};
}
function torsoLoc(c, face){
  const {w,d}=torsoProfileAt(c.t);
  const r=Math.max(-1,Math.min(1, c.off/c.halfW));
  const sx = face.front? 1 : -1;
  const toward=Math.sqrt(Math.max(0,1-r*r));
  return {segment:"torso", t:c.t,
          radial:norm([sx*r*w, 0, (face.front?1:-1)*toward*d])};
}

// ---------- head -------------------------------------------------------------
function headCandidate(tap, pose, ppm){
  const P=pose.points, nose=P[LM.nose];
  if(nose.v<VISIBLE) return null;
  const R=0.105*ppm;
  // The nose sits on the FRONT of the head, not at its centre; step back along
  // the model's own offset so a tap on the forehead is not read as off-body.
  const c={x:nose.x, y:nose.y-0.15*R};
  const off=[tap.x-c.x, tap.y-c.y], dist=l2(off);
  return {kind:"head", id:"head", off2:off, R, dist, cost:dist/R};
}
function headLoc(c, face){
  const sx=face.front?1:-1, zc=face.front?1:-1;
  const rx=Math.max(-1,Math.min(1,c.off2[0]/c.R));
  const ry=Math.max(-1,Math.min(1,c.off2[1]/c.R));
  const toward=Math.sqrt(Math.max(0,1-Math.min(1,rx*rx+ry*ry)));
  return {segment:"head", t:0, radial:norm([sx*rx, -ry, zc*toward])};
}
