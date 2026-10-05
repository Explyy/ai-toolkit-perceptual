'use client';
import React, { memo, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { Image } from './store';
import { imageOpacity, MembershipFilter } from './gallerySelection';
export type CardDraft = {caption: string; revision: number; draftRevision: number; persisted?: string};
type Props = {dataset: string; image: Image; draft?: CardDraft; height: number; membership: MembershipFilter;
 pending: boolean; hit: boolean; tentative?: boolean;
 onToggle: (id: string) => void; onCategory: (id: string, category: string) => void;
 onCaption: (image: Image, caption: string) => void; onDiscardDraft: (id: string) => void};
function Preview({dataset,image,height,membership}:{dataset:string;image:Image;height:number;membership:MembershipFilter}) {
 const holder=useRef<HTMLDivElement>(null),[url,setUrl]=useState(''),[failed,setFailed]=useState(false),[attempt,setAttempt]=useState(0);
 useEffect(()=>{
  let dead=false,created='',started=false;
  const observer=new IntersectionObserver(entries=>{
   if(started||!entries.some(x=>x.isIntersecting))return;started=true;observer.disconnect();setFailed(false);
   apiClient.get('/api/dataset-studio',{params:{dataset,image:image.id},responseType:'blob'}).then(r=>{
    if(dead)return;created=URL.createObjectURL(r.data);setUrl(created);
   }).catch(()=>{if(!dead)setFailed(true);});
  },{rootMargin:'400px'});
  if(holder.current)observer.observe(holder.current);
  return()=>{dead=true;observer.disconnect();if(created)URL.revokeObjectURL(created);};
 },[dataset,image.id,attempt]);
 return <div ref={holder} style={{height}} className="w-full flex items-center justify-center rounded bg-gray-950">
  {url?<img src={url} draggable={false} alt={image.filename} style={{opacity:imageOpacity(image.excluded,membership),width:'100%',height:'100%',maxWidth:'100%',maxHeight:height,minHeight:0,objectFit:'contain'}} />:
   failed?<button onClick={()=>setAttempt(x=>x+1)}>Carica di nuovo l’immagine</button>:<span className="text-gray-400">Caricamento…</span>}
 </div>;
}
function Card({dataset,image,draft,height,membership,pending,hit,tentative,onToggle,onCategory,onCaption,onDiscardDraft}:Props){
 return <article className="min-w-0 rounded-xl space-y-3" style={{minWidth:0,padding:12,background:'#111827',border:hit?'2px solid #60a5fa':'2px solid transparent'}}>
  <div style={{display:'flex',flexWrap:'wrap',alignItems:'center',gap:8,minWidth:0}}>
   <input type="checkbox" aria-label={'Seleziona '+image.filename} className="h-5 w-5" checked={!image.excluded} onChange={()=>onToggle(image.id)}/>
   <span title={image.filename} style={{flex:'1 1 120px',minWidth:0,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{image.filename}</span>
   {image.analysisStatus?.phase==='complete'&&<select aria-label={'Categoria '+image.filename} className="rounded border border-gray-600 bg-gray-950 p-1" value={image.category} onChange={e=>onCategory(image.id,e.target.value)}>
    <option value="face">Volto</option><option value="body">Corpo</option><option value="variety">Altre foto</option><option value="unclassified">Da analizzare</option>
   </select>}
  </div>
  <div data-selection-image={image.id} style={{cursor:'crosshair'}} onClick={e=>{
   if((e.nativeEvent as PointerEvent).pointerType==='mouse'||(e.target as Element).closest('button'))return;onToggle(image.id);
  }}><Preview dataset={dataset} image={image} height={height} membership={membership}/></div>
  <div className="flex flex-wrap gap-2 text-xs text-gray-400" aria-label={'Qualità '+image.filename}>
   <span>{image.analysis&&Number.isFinite(image.analysis.quality)?`Qualità ${Math.round(image.analysis.quality)}/100`:'Da analizzare'}</span>
   {image.categorySource==='manual'&&<span>Categoria rivista</span>}
   {pending&&<span className="text-blue-300">Da salvare</span>}
   {tentative!==undefined&&<span className="text-amber-200">{image.analysisStatus?.phase!=='complete'?'Analisi in attesa':tentative?'Proposta: includi':'Proposta: escludi'}</span>}
  </div>
  <label className="block text-sm">Caption<textarea aria-label={'Caption '+image.filename} className="w-full min-w-0 rounded-lg border border-gray-600 bg-gray-950 p-3 mt-2 text-base" rows={4} value={draft?.caption??image.caption} onChange={e=>onCaption(image,e.target.value)}/></label>
  {draft&&<div className="flex items-center gap-3 text-sm"><span className="text-amber-200">{draft.persisted===draft.caption?'Bozza salvata':'Modifica da salvare'}</span><button className="underline" onClick={()=>onDiscardDraft(image.id)}>Scarta modifica</button></div>}
 </article>;
}
// Server responses recreate image objects. Compare only rendered fields so an
// unrelated membership response does not rerender or reload every photo.
export default memo(Card,(a,b)=>a.dataset===b.dataset&&a.image.id===b.image.id&&a.image.filename===b.image.filename&&
 a.image.excluded===b.image.excluded&&a.image.category===b.image.category&&a.image.categorySource===b.image.categorySource&&
 a.image.analysis?.quality===b.image.analysis?.quality&&a.image.analysisStatus?.phase===b.image.analysisStatus?.phase&&a.image.caption===b.image.caption&&a.image.revision===b.image.revision&&
 a.draft===b.draft&&a.height===b.height&&a.membership===b.membership&&a.pending===b.pending&&a.hit===b.hit&&a.tentative===b.tentative&&
 a.onToggle===b.onToggle&&a.onCategory===b.onCategory&&a.onCaption===b.onCaption&&a.onDiscardDraft===b.onDiscardDraft);
