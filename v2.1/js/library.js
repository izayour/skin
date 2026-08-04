// ---------- the two image libraries ---------------------------------------
// Deliberately two separate stores:
//
//   originals  -- every image brought into the app, byte for byte as it was
//                 imported. Nothing ever writes back into this store: whatever
//                 processing happens, the original stays intact. It stands in
//                 for "any folder" the user pulled the photo from.
//
//   processed  -- at most ONE record per original, holding the image as it
//                 stands after every step applied to it so far, plus which
//                 steps those were and when. Its id IS the original's id, so
//                 the pair is 1:1 and either half of a lineage is a direct
//                 lookup rather than a scan.
//
// IndexedDB, not localStorage: a working copy has to stay full-resolution to be
// fed into the next step, and a single phone photo already exceeds the ~5 MB
// localStorage budget on its own.
//
// A processed record carries three kinds of image, for three different jobs:
//   work     -- Blob of the pixels the NEXT step should read (only steps that
//               actually alter pixels, e.g. colour correction, replace it)
//   renders  -- one data URL PER STEP, showing what that step produced. A
//               single rolling preview was useless: each step overwrote the
//               last, so a lineage with four steps done showed only the fourth.
//   preview  -- the newest render, for the gallery tile
// Keeping work apart from the renders is what stops a later step from measuring
// an image with green outlines and cyan tick markers burnt into it.

const LIB_DB="skinlib", LIB_VER=1, LIB_THUMB=520;

// Steps a lineage can accumulate. The key is what the pickers gate on: an
// image is offered for a step only while its lineage lacks that key.
const LIB_STEPS=[
  ["colour", "Colour calibration", "the card's colour strips were located"],
  ["lesion", "Lesion outlined",    "border detected automatically or traced by hand"],
  ["ruler",  "Scale established",  "pixels-per-centimetre read from the ruler"],
  ["area",   "Size measured",      "diameter and area reported in millimetres"]
];

let _libDb=null;
function libDB(){
  if(_libDb) return Promise.resolve(_libDb);
  return new Promise((res,rej)=>{
    const rq=indexedDB.open(LIB_DB,LIB_VER);
    rq.onupgradeneeded=()=>{
      const d=rq.result;
      if(!d.objectStoreNames.contains("originals")) d.createObjectStore("originals",{keyPath:"id"});
      if(!d.objectStoreNames.contains("processed")) d.createObjectStore("processed",{keyPath:"id"});
    };
    rq.onsuccess=()=>{_libDb=rq.result;res(_libDb);};
    rq.onerror=()=>rej(rq.error);
  });
}
// Resolve on transaction *complete*, not on request success -- a value read out
// of a transaction that later aborts is a value that was never really there.
function libReq(stores,mode,fn){
  return libDB().then(d=>new Promise((res,rej)=>{
    const tx=d.transaction(stores,mode);
    let val;
    const rq=fn(Array.isArray(stores)
      ? Object.fromEntries(stores.map(s=>[s,tx.objectStore(s)]))
      : tx.objectStore(stores));
    if(rq&&"onsuccess" in rq) rq.onsuccess=()=>{val=rq.result;};
    tx.oncomplete=()=>res(val);
    tx.onerror=()=>rej(tx.error);
    tx.onabort=()=>rej(tx.error);
  }));
}
const libGet=(store,id)=>libReq(store,"readonly",st=>st.get(id));
const libAll=store=>libReq(store,"readonly",st=>st.getAll());
const libPut=(store,rec)=>libReq(store,"readwrite",st=>st.put(rec)).then(()=>rec);

function libBlobToImage(blob){
  return new Promise((res,rej)=>{
    const u=URL.createObjectURL(blob), im=new Image();
    // safe to revoke on load: the decoded bitmap outlives the object URL
    im.onload=()=>{URL.revokeObjectURL(u);res(im);};
    im.onerror=e=>{URL.revokeObjectURL(u);rej(e);};
    im.src=u;
  });
}
function libThumb(src,w,h,max){
  const s=Math.min(1,(max||LIB_THUMB)/Math.max(w,h));
  const t=document.createElement("canvas");
  t.width=Math.max(1,Math.round(w*s)); t.height=Math.max(1,Math.round(h*s));
  t.getContext("2d").drawImage(src,0,0,t.width,t.height);
  return t.toDataURL("image/jpeg",0.72);
}

// ---- originals ------------------------------------------------------------
async function libImport(blob,name){
  const im=await libBlobToImage(blob);
  const rec={id:String(Date.now())+"-"+Math.random().toString(36).slice(2,7),
             name:name||"(photo)", addedAt:new Date().toISOString(),
             w:im.naturalWidth, h:im.naturalHeight, bytes:blob.size,
             blob, thumb:libThumb(im,im.naturalWidth,im.naturalHeight)};
  await libPut("originals",rec);
  return rec;
}

// ---- processed ------------------------------------------------------------
// Marking a step creates the processed record on first use, seeded from the
// original -- so an image only enters the processed library once something has
// actually been done to it.
//   opts.render   data URL of what THIS step produced; filed under the step's
//                 own key and also promoted to the gallery preview
//   opts.work     Blob replacing the pixels later steps read (+ opts.w/h)
//   opts.meta     merged into the record, for numbers worth keeping
// Every mark is a read-modify-write across two awaits, so two of them in flight
// at once (finalize() marks "ruler" and "area" back to back) both read the same
// record and the second write drops the first's step. One queue, no races.
let _libQueue=Promise.resolve();
function libMarkStep(id,step,opts){
  const run=()=>_libMarkStep(id,step,opts);
  const next=_libQueue.then(run,run);
  _libQueue=next.catch(()=>{});      // a failed mark must not wedge the queue
  return next;
}
async function _libMarkStep(id,step,opts){
  opts=opts||{};
  const [orig,existing]=await Promise.all([libGet("originals",id),libGet("processed",id)]);
  if(!orig&&!existing) return null;
  const p=existing||{id, name:orig.name, addedAt:orig.addedAt,
                     createdAt:new Date().toISOString(),
                     w:orig.w, h:orig.h, work:orig.blob, preview:orig.thumb,
                     steps:{}, renders:{}, meta:{}};
  if(!p.renders) p.renders={};
  p.steps[step]=new Date().toISOString();
  p.updatedAt=new Date().toISOString();
  if(opts.render){ p.renders[step]=opts.render; p.preview=opts.render; }
  if(opts.work){ p.work=opts.work; if(opts.w) p.w=opts.w; if(opts.h) p.h=opts.h; }
  if(opts.meta) p.meta=Object.assign({},p.meta,opts.meta);
  await libPut("processed",p);
  return p;
}

// Full-resolution pixels to feed the next step: the processed working copy when
// one exists, otherwise the untouched original.
async function libWorkBlob(id){
  const p=await libGet("processed",id);
  if(p&&p.work) return p.work;
  const o=await libGet("originals",id);
  return o?o.blob:null;
}

// ---- choosing --------------------------------------------------------------
// One entry per original, showing the most processed version of it. An entry
// drops out once `step` has been applied to that lineage -- which is the whole
// point: pressing "Colour calibration" offers only images that have not been
// colour calibrated. `includeDone` is the escape hatch for redoing one.
async function libCandidates(step,includeDone){
  const [os,ps]=await Promise.all([libAll("originals"),libAll("processed")]);
  const byId=Object.fromEntries(ps.map(p=>[p.id,p]));
  return os.map(o=>{
      const p=byId[o.id];
      return {id:o.id, name:o.name, addedAt:o.addedAt,
              w:(p||o).w, h:(p||o).h,
              thumb:p?p.preview:o.thumb,
              steps:p?p.steps:{}, processed:!!p};
    })
    .filter(e=>includeDone||!e.steps[step])
    .sort((a,b)=>new Date(b.addedAt)-new Date(a.addedAt));
}

// Dropping a processed record alone undoes every step and leaves the original
// available again; dropping the original takes the whole lineage with it.
function libForget(id,keepOriginal){
  return libDB().then(d=>new Promise((res,rej)=>{
    const tx=d.transaction(["originals","processed"],"readwrite");
    tx.objectStore("processed").delete(id);
    if(!keepOriginal) tx.objectStore("originals").delete(id);
    tx.oncomplete=()=>res();
    tx.onerror=()=>rej(tx.error);
  }));
}
