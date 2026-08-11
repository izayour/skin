// ---------- the canonical body, and the one table that defines it -----------
//
// Everything downstream reads SEGMENTS. The 3D mesh is built from it, the photo
// is matched against it, and a stored location is meaningless without it. That
// is deliberate: the moment the mesh and the matcher carry their own idea of
// where a forearm is, a marker drawn on the model stops meaning the same thing
// as the spot that was measured, and nothing in the UI would reveal the drift.
//
// Frame: +X = the SUBJECT'S OWN LEFT, +Y = up, +Z = the direction they face.
// Metres, on a 1.75 m adult. So a camera parked on +Z sees them face on, with
// their left hand toward the right of the screen -- the same way they appear in
// a photo taken of them, which is what makes the photo->model mapping direct.
//
// Each limb segment also carries the MediaPipe landmark pair that stands for it
// in a photograph (`lm`). That pairing is the whole bridge between the two
// worlds: two points in the image, two points in the model, one segment.

export const JOINTS = {
  head:       [ 0,     1.605, 0.010],
  neckTop:    [ 0,     1.500, 0    ],
  neckBase:   [ 0,     1.430, 0    ],
  shoulderC:  [ 0,     1.400, 0    ],
  hipC:       [ 0,     0.950, 0    ],

  shoulder_l: [ 0.185, 1.400, 0    ], shoulder_r: [-0.185, 1.400, 0    ],
  elbow_l:    [ 0.255, 1.110, 0    ], elbow_r:    [-0.255, 1.110, 0    ],
  wrist_l:    [ 0.315, 0.830, 0    ], wrist_r:    [-0.315, 0.830, 0    ],
  handTip_l:  [ 0.345, 0.700, 0    ], handTip_r:  [-0.345, 0.700, 0    ],

  hip_l:      [ 0.095, 0.950, 0    ], hip_r:      [-0.095, 0.950, 0    ],
  knee_l:     [ 0.105, 0.500, 0    ], knee_r:     [-0.105, 0.500, 0    ],
  ankle_l:    [ 0.100, 0.080, 0    ], ankle_r:    [-0.100, 0.080, 0    ],
  toe_l:      [ 0.100, 0.035, 0.140], toe_r:      [-0.100, 0.035, 0.140],
};

// MediaPipe Pose landmark indices worth naming (see BodyTrack/body_parts_reference.md)
export const LM = {
  nose:0, ear_l:7, ear_r:8,
  shoulder_l:11, shoulder_r:12, elbow_l:13, elbow_r:14, wrist_l:15, wrist_r:16,
  index_l:19, index_r:20,
  hip_l:23, hip_r:24, knee_l:25, knee_r:26, ankle_l:27, ankle_r:28,
  toe_l:31, toe_r:32,
};

// `near`/`far` name the ends so a position along a limb can be reported the way
// a person would say it ("two thirds down toward the wrist") rather than as t.
const limb = (id, label, a, b, r0, r1, lmA, lmB, near, far, side) =>
  ({id, kind:"limb", label, a, b, r0, r1, lm:[lmA,lmB], near, far, side});

export const SEGMENTS = [
  // ---- arms
  limb("upperarm_l","left upper arm","shoulder_l","elbow_l",0.050,0.043,LM.shoulder_l,LM.elbow_l,"shoulder","elbow","left"),
  limb("upperarm_r","right upper arm","shoulder_r","elbow_r",0.050,0.043,LM.shoulder_r,LM.elbow_r,"shoulder","elbow","right"),
  limb("forearm_l","left forearm","elbow_l","wrist_l",0.043,0.031,LM.elbow_l,LM.wrist_l,"elbow","wrist","left"),
  limb("forearm_r","right forearm","elbow_r","wrist_r",0.043,0.031,LM.elbow_r,LM.wrist_r,"elbow","wrist","right"),
  limb("hand_l","left hand","wrist_l","handTip_l",0.032,0.026,LM.wrist_l,LM.index_l,"wrist","fingers","left"),
  limb("hand_r","right hand","wrist_r","handTip_r",0.032,0.026,LM.wrist_r,LM.index_r,"wrist","fingers","right"),
  // ---- legs
  limb("thigh_l","left thigh","hip_l","knee_l",0.088,0.058,LM.hip_l,LM.knee_l,"hip","knee","left"),
  limb("thigh_r","right thigh","hip_r","knee_r",0.088,0.058,LM.hip_r,LM.knee_r,"hip","knee","right"),
  limb("shin_l","left lower leg","knee_l","ankle_l",0.058,0.040,LM.knee_l,LM.ankle_l,"knee","ankle","left"),
  limb("shin_r","right lower leg","knee_r","ankle_r",0.058,0.040,LM.knee_r,LM.ankle_r,"knee","ankle","right"),
  limb("foot_l","left foot","ankle_l","toe_l",0.042,0.035,LM.ankle_l,LM.toe_l,"ankle","toes","left"),
  limb("foot_r","right foot","ankle_r","toe_r",0.042,0.035,LM.ankle_r,LM.toe_r,"ankle","toes","right"),
  // ---- neck: no landmark pair of its own, so it is matched off the head-to-
  // shoulders axis rather than off two joints (see locate.js)
  {id:"neck", kind:"limb", label:"neck", a:"neckBase", b:"neckTop", r0:0.058, r1:0.052,
   lm:null, near:"shoulders", far:"jaw", side:"centre"},
  // ---- torso and head are not capsules; they carry their own parameterisation
  {id:"torso", kind:"torso", label:"torso", side:"centre",
   top:1.400, bottom:0.950,
   // (v, halfWidth, halfDepth) -- v runs 0 at the shoulder line to 1 at the hips
   profile:[[0.00,0.185,0.105],[0.30,0.170,0.100],[0.55,0.148,0.090],
            [0.80,0.155,0.095],[1.00,0.165,0.100]]},
  {id:"head", kind:"head", label:"head", side:"centre", centre:"head", r:0.105},
];

export const SEG = Object.fromEntries(SEGMENTS.map(s=>[s.id,s]));

// ---------- vector helpers (kept plain: three.js is not loaded on the photo
// side of the app, and this table has to be readable from both) --------------
export const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
export const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
export const mul=(a,k)=>[a[0]*k,a[1]*k,a[2]*k];
export const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const len=a=>Math.hypot(a[0],a[1],a[2]);
export const norm=a=>{const l=len(a)||1;return [a[0]/l,a[1]/l,a[2]/l];};
export const lerp3=(a,b,t)=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

export const jointOf = name => JOINTS[name];
export const segAxis = s => norm(sub(JOINTS[s.b],JOINTS[s.a]));
export const segLength = s => len(sub(JOINTS[s.b],JOINTS[s.a]));
export const radiusAt = (s,t) => s.r0+(s.r1-s.r0)*Math.min(1,Math.max(0,t));

// Torso half-width / half-depth at a given v, read off the profile.
export function torsoProfileAt(v){
  const p=SEG.torso.profile;
  v=Math.min(1,Math.max(0,v));
  for(let i=1;i<p.length;i++){
    if(v<=p[i][0]){
      const [v0,w0,d0]=p[i-1], [v1,w1,d1]=p[i], k=(v-v0)/((v1-v0)||1);
      return {w:w0+(w1-w0)*k, d:d0+(d1-d0)*k};
    }
  }
  return {w:p[p.length-1][1], d:p[p.length-1][2]};
}

// ---------- a stored location, resolved back to a point in space -------------
// A location is {segment, t, radial}: which segment, how far along it, and the
// outward direction at that spot. Storing the direction rather than a raw XYZ
// is what lets the model be re-proportioned later without the saved marks
// sliding off the body.
export function surfacePoint(loc, lift=1.0){
  const s=SEG[loc.segment];
  if(!s) return {point:[0,1,0], normal:[0,0,1]};

  if(s.kind==="head"){
    const c=JOINTS[s.centre], n=norm(loc.radial);
    return {point:add(c,mul(n,s.r*lift)), normal:n};
  }
  if(s.kind==="torso"){
    // radial is stored as a direction; intersect it with the elliptical cross
    // section at height v to land exactly on the surface.
    const v=loc.t, {w,d}=torsoProfileAt(v);
    const y=s.top+(s.bottom-s.top)*v;
    const n=norm([loc.radial[0],0,loc.radial[2]]);
    const k=1/Math.hypot(n[0]/w, n[2]/d || 1e-6);
    const p=[n[0]*k*lift, y, n[2]*k*lift];
    // the true outward normal of an ellipse is not the radius direction
    return {point:p, normal:norm([p[0]/(w*w),0,p[2]/(d*d)])};
  }
  const A=JOINTS[s.a], B=JOINTS[s.b];
  const c=lerp3(A,B,loc.t), r=radiusAt(s,loc.t);
  const n=norm(loc.radial);
  return {point:add(c,mul(n,r*lift)), normal:n};
}

// ---------- saying it out loud ----------------------------------------------
// The point of the whole feature is a sentence a person can act on -- read back
// at a follow-up visit, or repeated to a clinician. "0.62 along forearm_l" is
// not that sentence.
export function describe(loc){
  const s=SEG[loc.segment];
  if(!s) return "unknown location";
  if(s.kind==="head") return "head";

  const aspect = aspectWord(loc, s);
  if(s.kind==="torso"){
    const band = loc.t<0.22 ? "upper chest" : loc.t<0.45 ? "chest"
               : loc.t<0.72 ? "midriff" : "lower abdomen";
    const front = loc.radial[2]>=0;
    const side  = Math.abs(loc.radial[0])<0.35 ? "centre"
                : (loc.radial[0]>0 ? "left" : "right");
    const name  = front ? band : (loc.t<0.45?"upper back":loc.t<0.72?"mid back":"lower back");
    return side==="centre" ? name : `${name}, ${side} side`;
  }
  const pct = Math.round(loc.t*100);
  const where = pct<=12 ? `at the ${s.near}`
              : pct>=88 ? `at the ${s.far}`
              : `${pct}% of the way from ${s.near} to ${s.far}`;
  return `${s.label}, ${where}${aspect?`, ${aspect}`:""}`;
}

// Which face of the limb it sits on. Lateral/medial is relative to the body's
// midline, so it flips sign between the left and right sides -- getting that
// backwards would send someone looking at the wrong side of their own arm.
function aspectWord(loc, s){
  if(s.kind!=="limb") return "";
  const [x,,z]=norm(loc.radial);
  if(Math.abs(z)>0.65) return z>0 ? "front" : "back";
  if(Math.abs(x)>0.65){
    const outward = s.side==="left" ? x>0 : x<0;
    return outward ? "outer side" : "inner side";
  }
  const fb = z>0?"front":"back";
  const outward = s.side==="left" ? x>0 : x<0;
  return `${fb}-${outward?"outer":"inner"}`;
}
