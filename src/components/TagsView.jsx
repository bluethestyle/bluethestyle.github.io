import React from "react";
import { TAGS } from "../data.js";

// ========= TAGS TAB =========

function TagsView() {
  const tags = TAGS.slice().sort((a,b)=>b.count-a.count);
  const max = tags[0].count;
  return (
    <>
      <section className="hero" style={{paddingBottom:32, marginBottom:32}}>
        <div data-lang-ui="en">
          <div className="label"><span className="pulse"></span>Navigation · {tags.length} tags</div>
          <h1 style={{fontSize:52}}>
            <em>Tags</em> — the topics covered<br/>most often, sized by frequency.
          </h1>
          <p>Sized by frequency. Hot tags are currently driving most of the writing. Click any tag to filter the archive.</p>
        </div>
        <div data-lang-ui="ko">
          <div className="label"><span className="pulse"></span>내비게이션 · 태그 {tags.length}개</div>
          <h1 className="kr" style={{fontSize:52}}>
            <em>태그</em> — 자주 다루는 주제들,<br/>사이즈는 빈도.
          </h1>
          <p className="kr">빈도에 따라 크기가 결정된다. hot 태그는 최근 글쓰기를 주도하고 있다. 태그를 클릭하면 해당 태그의 글로 이동한다.</p>
        </div>
      </section>

      {/* cloud */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title" data-lang-ui="en">Tag <em>cloud</em></div>
            <div className="title kr" data-lang-ui="ko">태그 <em>구름</em></div>
            <div className="count" data-i18n-ko="글 수 기준 크기">sized by post count</div>
          </div>
          <div className="filters">
            <a className="on" data-i18n-ko="빈도순">by count</a><a data-i18n-ko="가나다순">a–z</a><a data-i18n-ko="최근순">recent</a>
          </div>
        </div>

        <div style={{background:"var(--surface)", border:"1px solid var(--hair)", borderRadius:10, padding:"40px 32px", display:"flex", flexWrap:"wrap", gap:"12px 14px", alignItems:"baseline"}}>
          {tags.map(t => {
            const scale = 0.7 + (t.count / max) * 1.1;
            return (
              <span key={t.slug} style={{
                fontFamily: "var(--display)",
                fontStyle: "italic",
                fontSize: `${18 * scale}px`,
                lineHeight: 1.1,
                letterSpacing: "-0.02em",
                color: t.hot ? "var(--ink)" : "var(--ink-2)",
                opacity: t.hot ? 1 : 0.65,
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "baseline",
                gap: 4,
                position: "relative",
              }}>
                #{t.name}
                <span style={{fontFamily:"JetBrains Mono", fontStyle:"normal", fontSize:10, color:"var(--muted-2)", marginLeft:2}}>{t.count}</span>
                {t.hot && <span style={{position:"absolute", top:-6, right:-8, width:5, height:5, borderRadius:"50%", background:"var(--accent)"}}></span>}
              </span>
            );
          })}
        </div>
      </section>

      {/* table */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title">All <em>tags</em></div>
            <div className="count">{tags.length} · ranked</div>
          </div>
        </div>
        <div style={{background:"var(--surface)", border:"1px solid var(--hair)", borderRadius:10, overflow:"hidden"}}>
          <div style={{display:"grid", gridTemplateColumns:"48px 1fr 120px 90px 80px", padding:"12px 20px", fontFamily:"JetBrains Mono", fontSize:10, letterSpacing:".12em", textTransform:"uppercase", color:"var(--muted)", borderBottom:"1px solid var(--hair)"}}>
            <span>#</span><span>tag</span><span>posts</span><span>share</span><span></span>
          </div>
          {tags.map((t,i) => (
            <div key={t.slug} style={{
              display:"grid",
              gridTemplateColumns:"48px 1fr 120px 90px 80px",
              padding:"14px 20px",
              alignItems:"center",
              borderBottom: i===tags.length-1 ? "0" : "1px solid var(--hair-2)",
              cursor:"pointer",
              fontFamily:"JetBrains Mono", fontSize:12, color:"var(--ink-2)"
            }}
            onMouseEnter={e=>e.currentTarget.style.background="var(--hair-2)"}
            onMouseLeave={e=>e.currentTarget.style.background=""}>
              <span style={{color:"var(--muted-2)"}}>{String(i+1).padStart(2,"0")}</span>
              <span style={{color:"var(--ink)", fontWeight:500, display:"flex", alignItems:"center", gap:8}}>
                #{t.name}
                {t.hot && <span style={{fontSize:9, padding:"1px 5px", background:"var(--accent-soft)", color:"var(--accent-ink)", borderRadius:3, letterSpacing:".1em"}}>HOT</span>}
              </span>
              <span>{t.count}</span>
              <span style={{display:"flex", alignItems:"center", gap:6}}>
                <span style={{flex:1, height:4, background:"var(--hair)", borderRadius:2, overflow:"hidden", maxWidth:60}}>
                  <span style={{display:"block", height:"100%", width: (t.count/max*100)+"%", background:"var(--accent)"}}></span>
                </span>
                <span style={{color:"var(--muted)"}}>{Math.round(t.count/max*100)}%</span>
              </span>
              <span style={{textAlign:"right", color:"var(--muted-2)"}}>↗</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

export default TagsView;
