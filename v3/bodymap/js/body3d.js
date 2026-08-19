// ---------- the 3D body, built from the same table the matcher reads ---------
// No downloaded mesh: every part here is generated from SEGMENTS, which means a
// mesh surface and a matched location cannot disagree about where the forearm
// is. It also means each piece of geometry knows which body part it is, so a
// click on the model resolves to the same {segment, t, radial} a photo does --
// that is what makes "correct it by hand" and "place it automatically" produce
// the same kind of record instead of two rival ones.

import * as THREE from "../vendor/three.module.min.js";
import {SEGMENTS, SEG, JOINTS, surfacePoint, torsoProfileAt,
        sub, add, mul, dot, len, norm, lerp3, radiusAt} from "./anatomy.js";

const V=(a)=>new THREE.Vector3(a[0],a[1],a[2]);

export class BodyView{
  constructor(canvas){
    this.canvas=canvas;
    this.renderer=new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
    this.renderer.setPixelRatio(Math.min(2,devicePixelRatio||1));
    this.scene=new THREE.Scene();
    this.camera=new THREE.PerspectiveCamera(32,1,0.05,50);
    this.target=new THREE.Vector3(0,0.92,0);
    this.sph={r:4.6, az:0, el:0.12};          // az 0 = looking at the front
    this.markers=[]; this.selected=null;
    this.onPick=null;

    this.scene.add(new THREE.HemisphereLight(0xffffff,0x6a7080,2.1));
    const key=new THREE.DirectionalLight(0xffffff,1.5); key.position.set(1.2,2.0,2.4);
    this.scene.add(key);
    const rim=new THREE.DirectionalLight(0xbcd4ff,0.8); rim.position.set(-1.6,1.0,-2.0);
    this.scene.add(rim);

    this.body=buildBody();
    this.scene.add(this.body);
    this.markerGroup=new THREE.Group(); this.scene.add(this.markerGroup);

    this.ray=new THREE.Raycaster();
    this._bindInput();
    this._resize();
    addEventListener("resize",()=>this._resize());
    this._tick();
  }

  // ---- input: a hand-rolled orbit. OrbitControls lives in three's examples
  // folder, a second file to vendor for drag-to-spin plus pinch -- not worth it.
  _bindInput(){
    const c=this.canvas;
    let drag=null, moved=0, pinch=null;
    const pts=new Map();
    c.style.touchAction="none";
    c.addEventListener("pointerdown",e=>{
      c.setPointerCapture(e.pointerId); pts.set(e.pointerId,e);
      if(pts.size===1){ drag={x:e.clientX,y:e.clientY}; moved=0; }
      else if(pts.size===2){ pinch=this._pinchDist(pts); drag=null; }
    });
    c.addEventListener("pointermove",e=>{
      if(!pts.has(e.pointerId)) return;
      pts.set(e.pointerId,e);
      if(pts.size===2&&pinch){
        const d=this._pinchDist(pts);
        this.sph.r=Math.max(1.6,Math.min(9,this.sph.r*(pinch/d)));
        pinch=d; return;
      }
      if(!drag) return;
      const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
      moved+=Math.abs(dx)+Math.abs(dy);
      this.sph.az-=dx*0.008;
      this.sph.el=Math.max(-1.15,Math.min(1.15,this.sph.el+dy*0.006));
      drag={x:e.clientX,y:e.clientY};
      this.spin=null;                       // a hand on it cancels any animation
    });
    const up=e=>{
      pts.delete(e.pointerId);
      if(pts.size<2) pinch=null;
      // A short press is a placement, a long drag is a rotation. Without the
      // threshold every attempt to spin the model ends by moving the marker.
      if(drag&&moved<6&&this.onPick) this._pick(e);
      if(!pts.size) drag=null;
    };
    c.addEventListener("pointerup",up);
    c.addEventListener("pointercancel",up);
    c.addEventListener("wheel",e=>{
      e.preventDefault();
      this.sph.r=Math.max(1.6,Math.min(9,this.sph.r*(1+Math.sign(e.deltaY)*0.09)));
    },{passive:false});
  }
  _pinchDist(pts){
    const [a,b]=[...pts.values()];
    return Math.hypot(a.clientX-b.clientX, a.clientY-b.clientY)||1;
  }
  _pick(e){
    const r=this.canvas.getBoundingClientRect();
    const p=new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1,
                              -((e.clientY-r.top)/r.height)*2+1);
    this.ray.setFromCamera(p,this.camera);
    const hits=this.ray.intersectObjects(this.body.children,true)
                       .filter(h=>h.object.userData.segment);
    if(!hits.length) return;
    const h=hits[0];
    const loc=pointToLocation(h.object.userData.segment,[h.point.x,h.point.y,h.point.z]);
    if(loc) this.onPick(loc);
  }

  // ---- markers
  setMarkers(list, selectedId){
    this.markerGroup.clear();
    this.markers=[];
    for(const m of list){
      const {point,normal}=surfacePoint(m.loc,1.0);
      const sel = m.id===selectedId;
      const g=new THREE.Group();
      const bead=new THREE.Mesh(new THREE.SphereGeometry(sel?0.028:0.020,20,14),
        new THREE.MeshStandardMaterial({color:sel?0xff3b30:0xff8a3d,
          emissive:sel?0x5a0d08:0x3a1a04, roughness:0.45}));
      g.add(bead);
      // A ring lying on the skin reads as "on the surface" from any angle; a
      // bare dot floating at the silhouette edge does not.
      const ring=new THREE.Mesh(new THREE.TorusGeometry(sel?0.055:0.040,0.006,8,36),
        new THREE.MeshBasicMaterial({color:sel?0xff3b30:0xffab63,
          transparent:true, opacity:0.9, depthTest:false}));
      ring.renderOrder=3;
      ring.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),V(normal).normalize());
      g.add(ring);
      g.position.copy(V(point)).addScaledVector(V(normal).normalize(),0.004);
      g.userData.id=m.id;
      this.markerGroup.add(g);
      this.markers.push(g);
    }
  }

  // Bring a mark into view instead of leaving the user to hunt for it -- half
  // the marks land on the back, where the default camera never looks.
  focus(loc){
    const {point,normal}=surfacePoint(loc,1.0);
    const n=V(normal).normalize();
    const az=Math.atan2(n.x,n.z);
    const el=Math.max(-0.9,Math.min(0.9,Math.asin(Math.max(-1,Math.min(1,n.y)))*0.6));
    this.spin={az, el, ty:point[1]*0.55+0.92*0.45, r:Math.max(2.0,this.sph.r*0.82)};
  }
  view(name){
    const az={front:0, back:Math.PI, left:Math.PI/2, right:-Math.PI/2}[name]??0;
    this.spin={az, el:0.10, ty:0.92, r:4.6};
  }
  setOpacity(o){
    this.body.traverse(m=>{ if(m.material){ m.material.transparent=o<1; m.material.opacity=o; }});
  }

  _resize(){
    const r=this.canvas.getBoundingClientRect();
    const w=Math.max(1,r.width), h=Math.max(1,r.height);
    this.renderer.setSize(w,h,false);
    this.camera.aspect=w/h; this.camera.updateProjectionMatrix();
  }
  _tick(){
    requestAnimationFrame(()=>this._tick());
    if(this.spin){
      const s=this.spin, k=0.14;
      // shortest way round, or a mark on the back sends the model the long way
      let d=((s.az-this.sph.az+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;
      this.sph.az+=d*k;
      this.sph.el+=(s.el-this.sph.el)*k;
      this.sph.r +=(s.r -this.sph.r )*k;
      this.target.y+=(s.ty-this.target.y)*k;
      if(Math.abs(d)<0.002&&Math.abs(s.el-this.sph.el)<0.002) this.spin=null;
    }
    const {r,az,el}=this.sph;
    this.camera.position.set(
      this.target.x + r*Math.cos(el)*Math.sin(az),
      this.target.y + r*Math.sin(el),
      this.target.z + r*Math.cos(el)*Math.cos(az));
    this.camera.lookAt(this.target);
    this.renderer.render(this.scene,this.camera);
  }
}

// ---------- geometry ---------------------------------------------------------
const SKIN=()=>new THREE.MeshStandardMaterial({color:0xd7b49a, roughness:0.72, metalness:0.0});

function buildBody(){
  const g=new THREE.Group();
  const mat=SKIN();
  for(const s of SEGMENTS){
    if(s.kind==="limb"){
      const A=JOINTS[s.a], B=JOINTS[s.b], d=sub(B,A), L=len(d);
      // Tapered: a cylinder is radiusBottom at its -Y end, and the orientation
      // below puts A at -Y, so bottom is A's radius.
      const cyl=new THREE.Mesh(new THREE.CylinderGeometry(s.r1,s.r0,L,20,1,true),mat);
      cyl.position.copy(V(lerp3(A,B,0.5)));
      cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),V(norm(d)));
      cyl.userData.segment=s.id;
      g.add(cyl);
      // Joint balls: they round off the ends and, more usefully, fill the gap
      // where two segments meet at an angle so there is no hole to click through.
      for(const [j,r] of [[A,s.r0],[B,s.r1]]){
        const b=new THREE.Mesh(new THREE.SphereGeometry(r,18,12),mat);
        b.position.copy(V(j)); b.userData.segment=s.id; g.add(b);
      }
    }else if(s.kind==="head"){
      const h=new THREE.Mesh(new THREE.SphereGeometry(s.r,28,20),mat);
      h.position.copy(V(JOINTS[s.centre]));
      h.scale.set(0.92,1.12,1.0);           // a sphere reads as a ball, not a head
      h.userData.segment=s.id; g.add(h);
    }else if(s.kind==="torso"){
      const t=new THREE.Mesh(torsoGeometry(s),mat);
      t.userData.segment=s.id; g.add(t);
    }
  }
  return g;
}

// Rings of elliptical cross section stacked down the torso, capped at both ends.
// The caps sit inside the shoulder and hip balls, so they are never seen -- they
// exist so the raycaster cannot shoot through the body and hit the far wall.
function torsoGeometry(s){
  const RINGS=26, SIDES=36, pos=[], idx=[];
  for(let i=0;i<=RINGS;i++){
    const v=i/RINGS, {w,d}=torsoProfileAt(v);
    const y=s.top+(s.bottom-s.top)*v;
    for(let k=0;k<SIDES;k++){
      const a=(k/SIDES)*Math.PI*2;
      pos.push(Math.sin(a)*w, y, Math.cos(a)*d);
    }
  }
  for(let i=0;i<RINGS;i++)
    for(let k=0;k<SIDES;k++){
      const a=i*SIDES+k, b=i*SIDES+(k+1)%SIDES, c=a+SIDES, e=b+SIDES;
      idx.push(a,c,b, b,c,e);
    }
  const capTop=pos.length/3;  pos.push(0,s.top,0);
  const capBot=pos.length/3;  pos.push(0,s.bottom,0);
  for(let k=0;k<SIDES;k++){
    idx.push(capTop, k, (k+1)%SIDES);
    const base=RINGS*SIDES;
    idx.push(capBot, base+(k+1)%SIDES, base+k);
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute("position",new THREE.Float32BufferAttribute(pos,3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

// ---------- a point on the mesh, back into a location -----------------------
// The inverse of surfacePoint(). Clicking the model and matching a photo have to
// produce the same shape of record, or a hand correction would silently become
// a different kind of thing than the automatic placement it replaced.
export function pointToLocation(segId, p){
  const s=SEG[segId];
  if(!s) return null;
  if(s.kind==="head"){
    const c=JOINTS[s.centre];
    return {segment:segId, t:0, radial:norm(sub(p,c))};
  }
  if(s.kind==="torso"){
    const v=Math.max(0,Math.min(1,(p[1]-s.top)/(s.bottom-s.top)));
    return {segment:segId, t:v, radial:norm([p[0],0,p[2]])};
  }
  const A=JOINTS[s.a], B=JOINTS[s.b], ax=sub(B,A), L=len(ax), u=norm(ax);
  const t=Math.max(0,Math.min(1, dot(sub(p,A),u)/L));
  const c=lerp3(A,B,t);
  let rad=sub(p,c);
  rad=sub(rad, mul(u, dot(rad,u)));
  if(len(rad)<1e-5) rad=[0,0,1];
  return {segment:segId, t, radial:norm(rad)};
}
