import React from "react";
import { CATEGORIES } from "../data.js";

// ========= CATEGORIES TAB =========

function CategoriesView({ posts = [] }) {
  const today = new Date().toISOString().slice(0, 10);
  const POSTS = posts.filter(p => p.draft || p.date <= today);
  const cats = CATEGORIES.map((c) => ({
    ...c,
    visibleCount: POSTS.filter(p => !p.draft && (p.cat === c.name || p.cat === c.ko)).length,
  }));
  return (
    <>
      <section className="hero" style={{paddingBottom:32, marginBottom:32}}>
        <div data-lang-ui="en">
          <div className="label"><span className="pulse"></span>Navigation · 4 categories</div>
          <h1 style={{fontSize:52}}>
            <em>Categories</em> — everything here is<br/>a <span className="hl">field note</span>, grouped.
          </h1>
          <p>Long-form work is organized into four categories. Each post usually exists as an English/Korean pair. Empty categories are listed so you can see what's coming.</p>
        </div>
        <div data-lang-ui="ko">
          <div className="label"><span className="pulse"></span>내비게이션 · 4개 카테고리</div>
          <h1 className="kr" style={{fontSize:52}}>
            <em>카테고리</em> — 이곳의 모든 글은<br/>주제별로 묶인 <span className="hl">현장 노트</span>다.
          </h1>
          <p className="kr">장문 작업은 네 개 카테고리로 분류된다. 대부분의 글은 영어/한국어 페어로 존재한다. 비어 있는 카테고리도 표시해서 앞으로의 계획을 보여준다.</p>
        </div>
      </section>

      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title" data-lang-ui="en">All <em>categories</em></div>
            <div className="title kr" data-lang-ui="ko">전체 <em>카테고리</em></div>
            <div className="count">{cats.length} <span data-i18n-ko="개">total</span> · {cats.filter(c=>c.visibleCount).length} <span data-i18n-ko="활성">active</span></div>
          </div>
          <div className="filters">
            <a className="on" data-i18n-ko="전체">all</a><a data-i18n-ko="활성">active</a><a data-i18n-ko="예정">planned</a>
          </div>
        </div>

        <div className="cats-grid">
          {cats.map((c,i) => {
            const cvar = "var(--c" + c.color + ")";
            const cs = "var(--c" + c.color + "s)";
            const active = c.visibleCount > 0;
            const Tag = active ? "a" : "div";
            const href = active ? `/categories/${c.slug}/` : undefined;
            return (
              <Tag key={c.slug} href={href} style={{
                display: "block",
                color: "inherit",
                textDecoration: "none",
                padding: "26px 28px",
                background: "var(--surface)",
                border: "1px solid var(--hair)",
                borderRadius: 10,
                position: "relative",
                opacity: active ? 1 : 0.6,
                cursor: active ? "pointer" : "default",
                transition: "transform .15s, box-shadow .15s"
              }} onMouseEnter={e=>{if(active){e.currentTarget.style.transform="translateY(-2px)";e.currentTarget.style.boxShadow="0 10px 30px rgba(0,0,0,0.06)";}}} onMouseLeave={e=>{e.currentTarget.style.transform="";e.currentTarget.style.boxShadow="";}}>
                <div style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:14}}>
                  <div style={{display:"inline-flex", alignItems:"center", gap:10}}>
                    <span style={{width:10, height:10, borderRadius:2, background:cvar}}></span>
                    <span style={{fontFamily:"JetBrains Mono", fontSize:10, letterSpacing:".12em", textTransform:"uppercase", color:"var(--muted)"}}>cat {String(i+1).padStart(2,"0")}</span>
                  </div>
                  <span style={{fontFamily:"JetBrains Mono", fontSize:10, color: active?"var(--ink-2)":"var(--muted-2)", padding:"2px 8px", border:"1px solid var(--hair)", borderRadius:3}}>
                    {active ? <><span>{c.visibleCount}</span> <span data-i18n-ko="글">posts</span></> : <span data-i18n-ko="비어있음">empty</span>}
                  </span>
                </div>
                <h3 data-lang-ui="en" style={{fontFamily:"var(--display)", fontSize:30, fontWeight:400, margin:"0 0 4px", letterSpacing:"-0.02em", lineHeight:1.05}}>{c.name}</h3>
                <h3 data-lang-ui="ko" className="kr" style={{fontFamily:"var(--display)", fontSize:30, fontWeight:400, margin:"0 0 4px", letterSpacing:"-0.02em", lineHeight:1.05}}>{c.ko}</h3>
                <div className="kr" data-lang-ui="en" style={{fontSize:14, color:"var(--muted)", marginBottom:14}}>{c.ko}</div>
                <div data-lang-ui="ko" style={{fontSize:14, color:"var(--muted)", marginBottom:14}}>{c.name}</div>
                <p data-lang-ui="en" style={{fontSize:13.5, color:"var(--ink-2)", margin:"0 0 18px", lineHeight:1.55, maxWidth:420}}>{c.desc}</p>
                <p data-lang-ui="ko" className="kr" style={{fontSize:13.5, color:"var(--ink-2)", margin:"0 0 18px", lineHeight:1.55, maxWidth:420}}>{c.descKo || c.desc}</p>
                <div style={{display:"flex", gap:5, flexWrap:"wrap"}}>
                  {POSTS.filter(p=>!p.draft && (p.cat===c.name || p.cat===c.ko)).slice(0,4).map((p,j)=>(
                    <span key={j} data-lang-post={(p.lang||"").toLowerCase()} style={{fontFamily:"JetBrains Mono", fontSize:10, padding:"3px 7px", background:cs, color:cvar, borderRadius:3, letterSpacing:".02em"}}>
                      {p.lang}·{p.date.slice(5)}
                    </span>
                  ))}
                </div>
              </Tag>
            );
          })}
        </div>
      </section>

      {/* distribution bar */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title" data-lang-ui="en"><em>Distribution</em></div>
            <div className="title kr" data-lang-ui="ko"><em>구성</em></div>
            <div className="count" data-i18n-ko="카테고리별 글 수">by category</div>
          </div>
        </div>
        <div style={{background:"var(--surface)", border:"1px solid var(--hair)", borderRadius:10, padding:22}}>
          <div style={{display:"flex", height:24, borderRadius:4, overflow:"hidden", marginBottom:14}}>
            {cats.filter(c=>c.visibleCount).map(c => (
              <div key={c.slug} style={{background:"var(--c"+c.color+")", flex:c.visibleCount, display:"flex", alignItems:"center", justifyContent:"center", fontFamily:"JetBrains Mono", fontSize:10, color:"#fff", letterSpacing:".08em"}}>
                {c.visibleCount}
              </div>
            ))}
          </div>
          <div style={{display:"flex", gap:16, flexWrap:"wrap"}}>
            {cats.filter(c=>c.visibleCount).map(c => (
              <div key={c.slug} style={{display:"inline-flex", alignItems:"center", gap:7, fontFamily:"JetBrains Mono", fontSize:11, color:"var(--muted)"}}>
                <span style={{width:8, height:8, background:"var(--c"+c.color+")", borderRadius:2}}></span>
                {c.name} · {c.visibleCount}
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

export default CategoriesView;
