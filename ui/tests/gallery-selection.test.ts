import test from 'node:test';
import assert from 'node:assert/strict';
import { ActionLane } from '../src/datasetStudio/actionLane';
import { galleryVisible, gallerySections, galleryCategories, imageOpacity, MembershipController, rebaseGalleryDrafts } from '../src/datasetStudio/gallerySelection';
const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
function deferred(){let resolve!:()=>void;const promise=new Promise<void>(r=>resolve=r);return{promise,resolve};}
const images=['face','body','variety','face','body','unclassified'].map((category,i)=>({id:String(i),filename:`shape-${i}.png`,caption:'geometric shape',category,excluded:i>=3?1:0,discarded:0,analysis:{quality:90-i*10},tags:['legacy preserved']}));
const filters={search:'',membership:'all' as const,categories:galleryCategories,minQuality:0};
test('selected groups precede excluded groups with stable category order and optional unclassified section',()=>{
 const sections=gallerySections(galleryVisible(images,{},filters));
 assert.deepEqual(sections.map(x=>[x.excluded,x.category,x.items.map(x=>x.id)]),[[0,'face',['0']],[0,'body',['1']],[0,'variety',['2']],[1,'face',['3']],[1,'body',['4']],[1,'unclassified',['5']]]);
 assert.equal(gallerySections(images.slice(0,3)).some(x=>x.category==='unclassified'),false);
});
test('membership/category/quality/search filters share exact visible set and excluded-only uses full opacity',()=>{
 const shown=galleryVisible(images,{'3':{caption:'draft portrait'}},{...filters,membership:'excluded',categories:['face','body'],minQuality:55,search:'portrait'});
 assert.deepEqual(shown.map(x=>x.id),['3']);
 assert.equal(imageOpacity(1,'excluded'),1);assert.equal(imageOpacity(1,'all'),.55);
 assert.equal(imageOpacity(0,'included'),1);
 assert.deepEqual(galleryVisible(images,{}, {...filters,categories:[]}),[]);
 assert.deepEqual(images[3].tags,['legacy preserved']);
});
function fixture(postDelay?:ReturnType<typeof deferred>){
 let state={revision:0,images:images.map(x=>({...x}))},valid=true;
 const lane=new ActionLane(),writes:{ids:string[];excluded:number;revision:number}[]=[],events:string[]=[];
 let delay=postDelay;
 const controller=new MembershipController({lane,key:'A:0:membership',valid:()=>valid,current:()=>state,
  post:async(ids,excluded)=>{events.push('post');const rev=state.revision;writes.push({ids:[...ids],excluded,revision:rev});if(delay){const d=delay;delay=undefined;await d.promise;}
   assert.equal(state.revision,rev);state={revision:rev+1,images:state.images.map(x=>ids.includes(x.id)?{...x,excluded}:x)};return state;},
  reconcile:async()=>{events.push('get');return state;},changed:()=>{}});
 return{controller,lane,writes,events,get state(){return state},setState:(s:typeof state)=>state=s,invalidate:()=>valid=false};
}
test('rapid delayed clicks remain optimistic, serialize through one lane and preserve newest intent after old response',async()=>{
 const d=deferred(),f=fixture(d);f.controller.set(['0'],1);await tick();
 f.controller.set(['0'],0);f.controller.set(['1'],1);
 assert.equal(f.state.images[0].excluded,0);assert.equal(f.controller.overlay(f.state.images)[0].excluded,0);
 assert.equal(f.controller.overlay(f.state.images)[1].excluded,1);assert.equal(f.controller.pending,2);
 assert.equal(f.lane.tryBackground(),null);assert.equal(f.writes.length,1);
 d.resolve();await f.controller.settled();
 assert.equal(f.state.images[0].excluded,0);assert.equal(f.state.images[1].excluded,1);
 assert.equal(f.controller.pending,0);assert.deepEqual(f.writes.map(x=>[x.ids,x.excluded]),[[['0'],1],[['0'],0],[['1'],1]]);
});
test('many intents queued behind a background save coalesce latest membership per identity',async()=>{
 const f=fixture(),release=f.lane.tryBackground()!;
 f.controller.set(['0'],1);f.controller.set(['0'],0);f.controller.set(['0','1'],1);f.controller.set(['1'],0);
 assert.equal(f.controller.overlay(f.state.images)[0].excluded,1);assert.equal(f.controller.overlay(f.state.images)[1].excluded,0);
 assert.equal(f.writes.length,0);release();await f.controller.settled();
 assert.deepEqual(f.writes.map(x=>[x.ids,x.excluded]),[[['1'],0],[['0'],1]]);
 assert.equal(f.controller.pending,0);
});
test('uncertain late original POST is fenced by a fresh revision CAS even when latest intent already matches GET',async()=>{
 let state={revision:0,images:[{id:'A',excluded:0}]},first=true,originalRevision=-1;const events:string[]=[];
 const c=new MembershipController({lane:new ActionLane(),key:'A:0',valid:()=>true,current:()=>state,
  post:async(ids,excluded)=>{events.push('post');if(first){first=false;originalRevision=state.revision;throw Error('lost request, may still be pending');}
   state={revision:state.revision+1,images:[{id:'A',excluded}]};return state;},
  reconcile:async()=>{events.push('get');return state;},changed:()=>{}});
 c.set(['A'],1);await c.settled();assert.ok(c.failed);assert.equal(c.overlay(state.images)[0].excluded,1);
 c.set(['A'],0);assert.equal(c.overlay(state.images)[0].excluded,0);assert.equal(events.length,1);
 await c.retry();assert.deepEqual(events,['post','get','post']);assert.equal(state.revision,1);assert.equal(state.images[0].excluded,0);
 assert.notEqual(state.revision,originalRevision,'Old guarded POST can no longer apply');assert.equal(c.pending,0);
});
test('lost applied response reconciles confirmed membership with zero repeated POSTs',async()=>{
 let state={revision:0,images:[{id:'A',excluded:0}]},posts=0,gets=0;
 const c=new MembershipController({lane:new ActionLane(),key:'A:0',valid:()=>true,current:()=>state,
  post:async(ids,excluded)=>{posts++;state={revision:1,images:[{id:'A',excluded}]};throw Error('response lost after commit');},
  reconcile:async()=>{gets++;return state;},changed:()=>{}});
 c.set(['A'],1);await c.settled();assert.equal(c.pending,1);await c.retry();
 assert.equal(posts,1);assert.equal(gets,1);assert.equal(c.pending,0);assert.equal(c.failed,null);
});
test('scope invalidation cancels queued old-dataset writes and its overlay cannot affect a fresh owner',async()=>{
 const f=fixture(),release=f.lane.tryBackground()!;f.controller.set(['0'],1);f.invalidate();release();await f.controller.settled();
 assert.equal(f.writes.length,0);const fresh=fixture();assert.equal(fresh.controller.pending,0);assert.equal(fresh.controller.overlay(fresh.state.images)[0].excluded,0);
});

import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import sharp from 'sharp';
import {StudioStore} from '../src/datasetStudio/store';
import {saveCaptionSnapshot} from '../src/datasetStudio/captionTextTools';
async function captionFixture(){
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'gallery-draft-owner-')),datasets=path.join(root,'datasets');
 const dir=path.join(datasets,'A');await fs.mkdir(dir,{recursive:true});
 await fs.writeFile(path.join(dir,'square.png'),await sharp({create:{width:8,height:8,channels:3,background:'#ee3344'}}).png().toBuffer());
 await fs.writeFile(path.join(dir,'square.txt'),'Original caption');
 const st=await new StudioStore(path.join(root,'data'),datasets,'A').init();let state=await st.read();
 const id=state.images[0].id;state=await st.captionDraft(state.revision,id,{caption:'Protected local draft',baseRevision:state.images[0].revision},0);
 const local={caption:'Protected local draft',revision:state.images[0].revision,draftRevision:state.images[0].captionDraftRevision!,persisted:'Protected local draft'};
 return{st,state,id,local,cleanup:()=>fs.rm(root,{recursive:true,force:true})};
}
test('reopened stale local caption stays stale through membership and category; newer caption cannot be overwritten',async()=>{
 for(const patch of [{excluded:1},{category:'body'}]){
  const f=await captionFixture();try{
   const newer=await f.st.edit(f.state.revision,[f.id],{caption:'Newer other-editor caption',baseRevision:f.state.images[0].revision});
   const reopened=await f.st.read(),image=reopened.images[0];
   const local={...f.local,caption:image.captionDraft!.caption,revision:image.captionDraft!.baseRevision,draftRevision:image.captionDraftRevision!};
   const saved=await f.st.edit(reopened.revision,[f.id],patch);
   const next=rebaseGalleryDrafts({[f.id]:local},[f.id],reopened.images,saved.images,'excluded' in patch)[f.id];
   assert.deepEqual(next,local,'Unrelated review must not legitimize a stale caption base');
   await assert.rejects(f.st.edit(saved.revision,[f.id],{caption:next.caption,baseRevision:next.revision}),/draft retained/);
   assert.equal((await f.st.read()).images[0].caption,'Newer other-editor caption');
  }finally{await f.cleanup();}
 }
});
test('fresh owned caption rebases through membership/category while preserving newer typed text and save CAS',async()=>{
 for(const patch of [{excluded:1},{category:'body'}]){
  const f=await captionFixture();try{
   const prior=f.state,latest={...f.local,caption:'Typed while response was pending'};
   const saved=await f.st.edit(prior.revision,[f.id],patch);
   const next=rebaseGalleryDrafts({[f.id]:latest},[f.id],prior.images,saved.images,'excluded' in patch)[f.id];
   assert.equal(next.caption,latest.caption);assert.equal(next.revision,saved.images[0].revision);
   assert.equal(next.draftRevision,saved.images[0].captionDraftRevision);assert.equal(next.persisted,undefined);
   let current=saved;
   await saveCaptionSnapshot(f.id,next,async(action,payload)=>current=action==='captionDraft'
    ?await f.st.captionDraft(current.revision,payload.id,payload.draft,payload.draftRevision)
    :await f.st.edit(current.revision,payload.ids,payload.patch),()=>{});
   const loaded=await f.st.read();assert.equal(loaded.images[0].caption,latest.caption);assert.equal(loaded.images[0].captionDraft,undefined);
  }finally{await f.cleanup();}
 }
});
test('competing protected token is never adopted by a stale local owner after unrelated selection',async()=>{
 const f=await captionFixture();try{
  const other=await f.st.captionDraft(f.state.revision,f.id,{caption:'Other tab protected text',baseRevision:f.state.images[0].revision},f.local.draftRevision);
  const saved=await f.st.edit(other.revision,[f.id],{excluded:1});
  const next=rebaseGalleryDrafts({[f.id]:f.local},[f.id],other.images,saved.images,true)[f.id];
  assert.deepEqual(next,f.local,'The local token does not own the observed protected draft');
  await assert.rejects(saveCaptionSnapshot(f.id,next,async(action,payload)=>action==='captionDraft'
   ?f.st.captionDraft(saved.revision,payload.id,payload.draft,payload.draftRevision)
   :f.st.edit(saved.revision,payload.ids,payload.patch),()=>{}),/nessun overwrite/);
  const loaded=await f.st.read();assert.equal(loaded.images[0].captionDraft!.caption,'Other tab protected text');assert.equal(loaded.images[0].caption,'Original caption');
 }finally{await f.cleanup();}
});
