// ---------- tying the close-up and the wide shot together, via the ruler ----
//
// Skin is nearly featureless and the two photos differ by 20-50x in scale, so
// matching the lesion itself across them is hopeless. The ruler is not: it is a
// flat, high-contrast, man-made rectangle, and it is already in both frames
// because the close-up needs it for scale anyway.
//
// Four corners of the same physical rectangle, seen twice, IS a homography --
// exactly determined, no descriptors, no scale invariance to fight. Push the
// lesion's centroid through it and its pixel in the wide shot falls out. The
// user taps the lesion where it is easy to hit (the close-up, where it fills
// the frame) and never has to hit a six-pixel dot at arm's length.
//
// Detection is NOT done here. An earlier version reimplemented it -- contour
// scoring, adaptive masks, a sweep of morphology kernels -- and lost to
// ruler.js on every count, because every global statistic it computed was
// dragged around by whatever else happened to be in the room. detectRuler
// works; it just has to be pointed at a box that is mostly skin with one card
// in it, which is the condition it was tuned for. Supplying those boxes is the
// caller's job (limbRois() in locate.js from the pose, belowLesionRoi() in the
// close-up, or a coarse tap from the user).
//
// What IS here is the part detectRuler does not do:
//
//  * CORRESPONDENCE. orderCorners() sorts corners in IMAGE space, so "top-left"
//    means a different physical corner once the phone is turned. Feeding
//    mismatched pairs to getPerspectiveTransform yields a twisted map that puts
//    the lesion confidently on the wrong side of the card. The colour swatch
//    strip runs along one long edge and is what breaks the 180-degree symmetry;
//    if it cannot be seen, this file says so rather than guessing.
//
//  * Deciding WHICH box's detection is really the ruler, and the local scale
//    the resulting map implies.

const _apply = (M,x,y) => {                 // M: 9 numbers, row-major
  const w = M[6]*x + M[7]*y + M[8];
  return [(M[0]*x + M[1]*y + M[2])/w, (M[3]*x + M[4]*y + M[5])/w];
};

export function matFromImage(img){
  const c=document.createElement("canvas");
  c.width=img.naturalWidth; c.height=img.naturalHeight;
  c.getContext("2d").drawImage(img,0,0);
  return cv.imread(c);
}

const rot = (p,n) => p.slice(n).concat(p.slice(0,n));
const dist = (a,b) => Math.hypot(a[0]-b[0], a[1]-b[1]);

function warpCard(src, corners, W, H){
  const s=cv.matFromArray(4,1,cv.CV_32FC2,
    [corners[0][0],corners[0][1], corners[1][0],corners[1][1],
     corners[2][0],corners[2][1], corners[3][0],corners[3][1]]);
  const d=cv.matFromArray(4,1,cv.CV_32FC2,[0,0, W-1,0, W-1,H-1, 0,H-1]);
  const M=cv.getPerspectiveTransform(s,d), out=new cv.Mat();
  cv.warpPerspective(src,out,M,new cv.Size(W,H));
  s.delete(); d.delete(); M.delete();
  return out;
}

// Mean saturation of the top third of the warped card against the bottom third.
// The swatch strip is the only strongly coloured thing on an otherwise white
// card, so this one number does two jobs: which way up the card is, and whether
// this rectangle is a card at all -- a doorway is bright but equally colourless
// top and bottom.
export function swatchContrast(src, corners){
  const long =(dist(corners[0],corners[1])+dist(corners[3],corners[2]))/2;
  const short=(dist(corners[1],corners[2])+dist(corners[0],corners[3]))/2;
  const W=240, H=Math.max(12,Math.round(240*short/(long||1)));
  const card=warpCard(src,corners,W,H);
  const rgb=new cv.Mat(); cv.cvtColor(card,rgb,cv.COLOR_RGBA2RGB);
  const hsv=new cv.Mat(); cv.cvtColor(rgb,hsv,cv.COLOR_RGB2HSV);
  const ch=new cv.MatVector(); cv.split(hsv,ch);
  const S=ch.get(1), V=ch.get(2);
  const band=(m,y0,y1)=>{const t=m.roi(new cv.Rect(0,y0,m.cols,Math.max(1,y1-y0)));
                         const v=cv.mean(t)[0]; t.delete(); return v;};
  const third=Math.max(1,Math.round(H/3));
  const satTop=band(S,0,third), satBot=band(S,H-third,H), val=cv.mean(V)[0];
  S.delete(); V.delete(); ch.delete(); hsv.delete(); rgb.delete(); card.delete();
  return {contrast:(satBot-satTop)/(satBot+satTop+1e-6), satTop, satBot, value:val};
}

// Long edge first, swatch strip along the bottom. Applied to both photos,
// corner i then means the same corner of the same card in both -- the entire
// precondition for the homography being meaningful rather than merely
// well-formed.
export function canonicalise(src, corners){
  let c=corners.map(p=>p.slice());
  const top =(dist(c[0],c[1])+dist(c[3],c[2]))/2;
  const side=(dist(c[1],c[2])+dist(c[0],c[3]))/2;
  if(side>top) c=rot(c,1);                       // make the long edge the top
  let s=swatchContrast(src,c);
  if(s.contrast<0){ c=rot(c,2); s=swatchContrast(src,c); }  // swatch was on top

  // How much contrast counts as certain depends on how big the card is on
  // screen. The swatch strip is a third of the card's width; at 330 px that is
  // 100 clean pixels to average, at 57 px it is 19 blurred ones. Measured on
  // the same physical card in both photos of one scene: 0.291 in the close-up,
  // 0.152 in the wide shot. Holding both to 0.18 rejected a wide-shot detection
  // whose corners were within 13 px of truth. Demand less of a small card, but
  // never less than a floor -- below that the sign itself stops being reliable,
  // and a wrong sign is a mark on the wrong face of the limb.
  const shortPx=Math.min(top,side);
  const need=Math.max(0.075, 0.18*Math.min(1, shortPx/120));
  return {corners:c, ...s, need, sure:Math.abs(s.contrast)>=need};
}

// ---------- narrowing a box to the skin inside it ---------------------------
// paperness (256-a-b) separates NEUTRAL from CHROMATIC, nothing more. Point it
// at a box that is half room and the room -- grey wall, grey carpet, grey
// daylight -- is every bit as "paper" as the card, the mask comes back 72%
// foreground as one blob, and maxAreaFrac throws the lot away. Measured, not
// guessed: that is precisely how a plainly visible ruler went missing.
//
// The cue that rescues it is that the card is lying ON someone. Skin is
// strongly chromatic and sits in a narrow, well-known band in YCrCb, so the
// skin's bounding box is a box the card is inside and the room is not -- and
// within it, neutral-versus-chromatic is once again the right question.
export function tightenToSkin(src, roi, opts){
  opts=Object.assign({minFrac:0.10, grow:0.06},opts||{});
  const rect=new cv.Rect(roi.x,roi.y,roi.w,roi.h);
  const t=src.roi(rect), crop=t.clone(); t.delete();
  const rgb=new cv.Mat(); cv.cvtColor(crop,rgb,cv.COLOR_RGBA2RGB);
  const ycc=new cv.Mat(); cv.cvtColor(rgb,ycc,cv.COLOR_RGB2YCrCb);
  const lo=new cv.Mat(ycc.rows,ycc.cols,ycc.type(),new cv.Scalar(0,133,77));
  const hi=new cv.Mat(ycc.rows,ycc.cols,ycc.type(),new cv.Scalar(255,180,127));
  const skin=new cv.Mat(); cv.inRange(ycc,lo,hi,skin);
  const k=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(9,9));
  cv.morphologyEx(skin,skin,cv.MORPH_OPEN,k,new cv.Point(-1,-1),1);
  const frac=cv.countNonZero(skin)/(skin.rows*skin.cols);
  let out=null;
  if(frac>=opts.minFrac){
    // cv.findNonZero is absent from this opencv.js build, so walk the mask.
    const d=skin.data, W=skin.cols, H=skin.rows;
    let x0=W, y0=H, x1=-1, y1=-1;
    for(let y=0;y<H;y++){
      const row=y*W;
      for(let x=0;x<W;x++){
        if(d[row+x]){
          if(x<x0)x0=x; if(x>x1)x1=x;
          if(y<y0)y0=y; if(y>y1)y1=y;
        }
      }
    }
    if(x1>=x0&&y1>=y0){
      // Grow a little: the card overhangs the limb rather than sitting wholly
      // within its silhouette.
      const g=Math.round(opts.grow*Math.max(roi.w,roi.h));
      const x=Math.max(0,roi.x+x0-g), y=Math.max(0,roi.y+y0-g);
      const w=Math.min(src.cols-x,(x1-x0+1)+2*g), h=Math.min(src.rows-y,(y1-y0+1)+2*g);
      if(w>=48&&h>=48) out={id:(roi.id||"")+" ∩skin", x, y, w, h};
    }
  }
  crop.delete(); rgb.delete(); ycc.delete(); lo.delete(); hi.delete();
  skin.delete(); k.delete();
  return {roi:out, skinFrac:frac};
}

// ---------- candidate quads inside one box ----------------------------------
// detectRuler pools EVERY qualifying contour and takes one minAreaRect around
// the union. In its own ROI -- skin, one card -- there is only ever one, so the
// pooling is invisible. In a limb box from a wide shot it is fatal: the card
// (measured 67x363, rectangularity 0.75, exactly right) got unioned with a
// sunlit doorway that fell in the same box, and out came a rectangle spanning
// the whole ROI.
//
// So: same cue, same cleanup -- paperness and cleanRuler are ruler.js's own
// globals and are called here directly -- but each contour is judged alone.
function quadsIn(src, roi, ksizes){
  const rect=new cv.Rect(roi.x,roi.y,roi.w,roi.h);
  const t=src.roi(rect), crop=t.clone(); t.delete();
  const rgb=new cv.Mat(); cv.cvtColor(crop,rgb,cv.COLOR_RGBA2RGB);
  const den=new cv.Mat(); cv.bilateralFilter(rgb,den,9,75,75,cv.BORDER_DEFAULT);
  const pap=paperness(den);
  const otsu=new cv.Mat();
  cv.threshold(pap,otsu,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU);
  const roiArea=roi.w*roi.h, out=[];
  for(const ks of ksizes){
    const cl=cleanRuler(otsu,ks|1);
    const cs=new cv.MatVector(), hi=new cv.Mat();
    cv.findContours(cl,cs,hi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE);
    for(let i=0;i<cs.size();i++){
      const c=cs.get(i), a=cv.contourArea(c), frac=a/roiArea;
      if(frac>=0.004&&frac<=0.55){
        const rr=cv.minAreaRect(c);
        const ra=rr.size.width*rr.size.height;
        const rectness=ra>0?a/ra:0;
        if(rectness>=0.5){
          const pts=cv.RotatedRect.points(rr)
                     .map(p=>[p.x+roi.x, p.y+roi.y]);
          out.push({corners:orderTLTRBRBL(pts), frac, rectness, ks});
        }
      }
      c.delete();
    }
    cs.delete(); hi.delete(); cl.delete();
  }
  crop.delete(); rgb.delete(); den.delete(); pap.delete(); otsu.delete();
  return out;
}

function orderTLTRBRBL(p){
  const s=p.map(q=>q[0]+q[1]), d=p.map(q=>q[1]-q[0]);
  const amin=a=>a.indexOf(Math.min(...a)), amax=a=>a.indexOf(Math.max(...a));
  return [p[amin(s)], p[amin(d)], p[amax(s)], p[amax(d)]];
}

// ---------- the ruler, in whichever box actually holds it -------------------
// Boxes arrive smallest first. A tight box is a better place to have found a
// card than a loose one, so the first convincing hit wins and the rest are not
// even tried; anything less than convincing is kept only as a runner-up.
export function findCardInRois(src, rois, opts){
  // detectRuler's default closeKsize of 25 is sized for a card that fills its
  // box. In a limb box from the pose the card may be a fifth of that, and the
  // OPEN pass then erodes it below the area floor and returns nothing at all --
  // which is exactly how a perfectly visible ruler went missing. The card's
  // apparent size is not known ahead of time, so try a few and take the best.
  // Each call is cheap because the box is small.
  opts=Object.assign({good:0.30, ksizes:[25,13,7], minAreaFrac:0.02},opts||{});
  const tried=[];
  let best=null;
  for(const raw of rois||[]){
    const tight=tightenToSkin(src,raw);
    const boxes=tight.roi?[tight.roi,raw]:[raw];
    for(const roi of boxes){
      for(const q of quadsIn(src,roi,opts.ksizes)){
        const can=canonicalise(src,q.corners);
        // Swatch contrast is what separates a ruler from any other pale
        // rectangle; rectangularity only says it is A rectangle.
        const score=Math.abs(can.contrast)+0.35*q.rectness;
        tried.push({roi:roi.id||"", ks:q.ks, frac:+q.frac.toFixed(3),
                    rectness:+q.rectness.toFixed(2),
                    contrast:+can.contrast.toFixed(3), score:+score.toFixed(3)});
        if(!best||score>best.score)
          best={corners:can.corners, contrast:can.contrast, sure:can.sure,
                roi, ks:q.ks, score, ticks:0, cmTicks:null};
      }
    }
    if(best&&best.score>=opts.good) break;
  }

  // Ticks last, and only for the winner: detectRuler aimed at a tight box round
  // the chosen quad, where its pooling has nothing left to pool.
  if(best){
    const xs=best.corners.map(p=>p[0]), ys=best.corners.map(p=>p[1]);
    const pad=0.15*Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys));
    const x=Math.max(0,Math.round(Math.min(...xs)-pad)), y=Math.max(0,Math.round(Math.min(...ys)-pad));
    const box={x, y, w:Math.min(src.cols-x,Math.round(Math.max(...xs)-Math.min(...xs)+2*pad)),
                     h:Math.min(src.rows-y,Math.round(Math.max(...ys)-Math.min(...ys)+2*pad))};
    if(box.w>=32&&box.h>=32){
      try{
        const r=detectRuler(src,box,{closeKsize:Math.max(5,Math.round(box.h/6))|1});
        if(r&&r.cmTicks){ best.ticks=r.cmTicks.length; best.cmTicks=r.cmTicks;
                          best.sharpness=r.sharpness; }
      }catch(err){ /* ticks are a bonus, not a requirement */ }
    }
  }
  window.__pairDiag=Object.assign(window.__pairDiag||{},{rois:tried});
  return best;
}

// ---------- the map itself ---------------------------------------------------
export function homography(fromCorners, toCorners){
  const s=cv.matFromArray(4,1,cv.CV_32FC2,fromCorners.flat());
  const d=cv.matFromArray(4,1,cv.CV_32FC2,toCorners.flat());
  const M=cv.getPerspectiveTransform(s,d);
  const out=Array.from(M.data64F);
  s.delete(); d.delete(); M.delete();
  return out;
}

// Local linear scale of the map at a point, from the determinant of its
// Jacobian. The ratio of card widths would be one number for the whole image,
// which is wrong the moment the card is tilted: near the lens a centimetre
// covers more pixels than it does at the far end of the same card.
export function scaleAt(M, x, y){
  const w=M[6]*x+M[7]*y+M[8];
  const u=(M[0]*x+M[1]*y+M[2])/w, v=(M[3]*x+M[4]*y+M[5])/w;
  const dux=(M[0]-u*M[6])/w, duy=(M[1]-u*M[7])/w;
  const dvx=(M[3]-v*M[6])/w, dvy=(M[4]-v*M[7])/w;
  return Math.sqrt(Math.abs(dux*dvy-duy*dvx));
}

export const mapPoint = (M,p) => _apply(M,p[0],p[1]);

// ---------- the whole association, end to end -------------------------------
// closePt is the lesion in close-up pixels; closeCard is what the close-up's
// own ruler detection produced; wideRois are the boxes to hunt in. Returns
// where the lesion lands in the wide shot plus the true pixels-per-cm THERE,
// which is worth as much as the point itself: it replaces the matcher's
// assumption that everyone is 1.75 m tall.
export function associate(closeMat, wideMat, closePt, closeCard, wideRois){
  const diag=window.__pairDiag={};
  if(!closeCard) return {ok:false, reason:"no ruler found in the close-up"};

  const cc=canonicalise(closeMat, closeCard.corners);
  diag.close={contrast:+cc.contrast.toFixed(3), sure:cc.sure};
  if(!cc.sure)
    return {ok:false, reason:"the colour strip is not clear enough in the close-up "+
                             "to tell which way round the ruler is"};

  const wide=findCardInRois(wideMat, wideRois);
  if(!wide) return {ok:false, reason:"no ruler found on the body in the wide shot"};
  diag.wide={contrast:+wide.contrast.toFixed(3), sure:wide.sure,
             roi:wide.roi.id||"", ticks:wide.ticks};
  if(!wide.sure)
    return {ok:false, reason:"the ruler is too small or too dim in the wide shot "+
                             "to tell which way round it is"};

  const M=homography(cc.corners, wide.corners);
  const p=mapPoint(M, closePt);
  if(!isFinite(p[0])||!isFinite(p[1])) return {ok:false, reason:"the two views do not line up"};
  if(p[0]<0||p[1]<0||p[0]>=wideMat.cols||p[1]>=wideMat.rows)
    return {ok:false, reason:"the lesion maps outside the wide shot — is it the same ruler?"};

  // A lesion metres from its own ruler means the corners were paired up wrongly;
  // better to fall back to a tap than to report that confidently.
  const cardLong=(dist(wide.corners[0],wide.corners[1])+
                  dist(wide.corners[3],wide.corners[2]))/2;
  const cx=wide.corners.reduce((s,q)=>s+q[0],0)/4;
  const cy=wide.corners.reduce((s,q)=>s+q[1],0)/4;
  const away=Math.hypot(p[0]-cx,p[1]-cy)/(cardLong||1);
  diag.awayCardLengths=+away.toFixed(2);
  if(away>3) return {ok:false, reason:"the lesion lands far from the ruler — corners may be mispaired"};

  const s=scaleAt(M, closePt[0], closePt[1]);
  return {ok:true, point:p, scale:s, closeCorners:cc.corners, wideCorners:wide.corners,
          pxPerCmWide: closeCard.pxPerCm ? closeCard.pxPerCm*s : null,
          roi:wide.roi, awayCardLengths:away};
}
