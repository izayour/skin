// ---------- finding the body in the wide photo ------------------------------
// MediaPipe Pose, run in the page against the vendored wasm. Nothing here is
// fetched from a CDN: the runtime, the wasm and the .task model all sit under
// bodymap/vendor, so the page keeps working on the phone over the LAN with no
// internet on the handset.

import {FilesetResolver, PoseLandmarker} from "../vendor/mediapipe/vision_bundle.mjs";

// One `..` fewer than the import above, and not a typo: an ES import resolves
// against THIS file (bodymap/js/), while MediaPipe fetches these two at runtime,
// which resolves against the document (bodymap/).
const WASM  = "./vendor/mediapipe/wasm";
const MODEL = "./vendor/mediapipe/models/pose_landmarker_lite.task";

let _lm=null, _loading=null;

export function poseReady(){ return !!_lm; }

export function loadPose(onProgress){
  if(_lm) return Promise.resolve(_lm);
  if(_loading) return _loading;
  _loading = (async()=>{
    onProgress&&onProgress("loading pose runtime…");
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    onProgress&&onProgress("loading pose model…");
    _lm = await PoseLandmarker.createFromOptions(fileset,{
      baseOptions:{modelAssetPath:MODEL, delegate:"GPU"},
      runningMode:"IMAGE",
      numPoses:1,
      // A wide shot of one limb is a *partial* body, which the default
      // thresholds throw away outright. Lowering them trades a few junk
      // landmarks -- which the visibility filter downstream drops anyway --
      // for actually getting a detection off a photo of just an arm.
      minPoseDetectionConfidence:0.25,
      minPosePresenceConfidence:0.25,
      minTrackingConfidence:0.25,
      outputSegmentationMasks:false,
    }).catch(async err=>{
      // Some Android GPUs reject the GPU delegate; CPU is slower but universal.
      console.warn("GPU delegate failed, retrying on CPU:",err);
      const fs2 = await FilesetResolver.forVisionTasks(WASM);
      return PoseLandmarker.createFromOptions(fs2,{
        baseOptions:{modelAssetPath:MODEL, delegate:"CPU"},
        runningMode:"IMAGE", numPoses:1,
        minPoseDetectionConfidence:0.25, minPosePresenceConfidence:0.25,
      });
    });
    return _lm;
  })();
  return _loading;
}

// Landmarks come back normalised to the image box; every consumer here wants
// pixels, so convert once at the boundary rather than at each use.
export async function detectPose(imgEl){
  const lm = await loadPose();
  const res = lm.detect(imgEl);
  if(!res.landmarks || !res.landmarks.length) return null;
  const w=imgEl.naturalWidth||imgEl.width, h=imgEl.naturalHeight||imgEl.height;
  const pts = res.landmarks[0].map(p=>({
    x:p.x*w, y:p.y*h, z:p.z,
    v: p.visibility!==undefined ? p.visibility : 1,
  }));
  return {points:pts, world:res.worldLandmarks?res.worldLandmarks[0]:null, w, h};
}

export const VISIBLE = 0.5;
export const seen = (pose,i) => pose && pose.points[i] && pose.points[i].v>=VISIBLE;

// Which way the subject is facing, which decides whether the camera is looking
// at the front or the back of every surface in the photo. MediaPipe labels
// landmarks anatomically, so the subject's own left appearing to the RIGHT in
// the image is exactly the mirror-image test: we are seeing them face on.
export function facing(pose){
  const P=pose.points;
  const pair = (a,b)=> P[a].v>=VISIBLE && P[b].v>=VISIBLE ? P[a].x-P[b].x : null;
  const d = pair(11,12) ?? pair(23,24) ?? pair(15,16) ?? pair(27,28);
  if(d===null) return {front:true, sure:false};
  return {front:d>0, sure:Math.abs(d)>4};
}

// Skeleton edges, for drawing the detection back over the photo. Showing the
// stick figure is not decoration -- it is the only way to tell "the mapping is
// wrong" apart from "the pose was wrong", which are fixed very differently.
export const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],
                      [11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],
                      [27,31],[28,32],[15,19],[16,20],[0,11],[0,12]];
