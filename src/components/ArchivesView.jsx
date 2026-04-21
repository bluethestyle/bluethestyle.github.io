import React from "react";

// ========= ARCHIVES TAB =========

function ArchivesView({ posts = [] }) {
  // group posts (all, including drafts) by year-month
  const all = posts.slice().sort((a,b)=>b.date.localeCompare(a.date));
  const byMonth = {};
  all.forEach(p => {
    const ym = p.date.slice(0,7);
    (byMonth[ym] = byMonth[ym] || []).push(p);
  });
  const months = Object.keys(byMonth).sort().reverse();

  // build calendar heat for 2026 Apr
  const heat = {};
  all.forEach(p => { heat[p.date] = (heat[p.date]||0) + 1; });

  return (
    <>
      <section className="hero" style={{paddingBottom:32, marginBottom:32}}>
        <div className="label"><span className="pulse"></span>Navigation · chronological</div>
        <h1 style={{fontSize:52}}>
          <em>Archives</em> — everything, in<br/>the order it was <span className="hl">written</span>.
        </h1>
        <p>A chronological index. Hollow dots are drafts; filled rows are published. The calendar shows writing density for the current month.</p>
      </section>

      {/* calendar heatmap */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title">April <em>2026</em></div>
            <div className="count">writing density</div>
          </div>
        </div>
        <div style={{background:"var(--surface)", border:"1px solid var(--hair)", borderRadius:10, padding:22}}>
          <div style={{display:"grid", gridTemplateColumns:"repeat(30, 1fr)", gap:4, maxWidth:720}}>
            {Array.from({length: 30}).map((_,i) => {
              const day = i + 1;
              const d = "2026-04-" + String(day).padStart(2,"0");
              const count = heat[d] || 0;
              const bg = count === 0 ? "var(--hair)" :
                         count === 1 ? "color-mix(in oklab, var(--accent) 35%, var(--bg))" :
                         count === 2 ? "color-mix(in oklab, var(--accent) 65%, var(--bg))" :
                                        "var(--accent)";
              return (
                <div key={i} title={d + " · " + count + " post(s)"} style={{
                  aspectRatio:"1/1", background: bg, borderRadius:2,
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontFamily:"JetBrains Mono", fontSize:8.5,
                  color: count ? "var(--bg)" : "var(--muted-2)",
                  cursor:"pointer"
                }}>
                  {day}
                </div>
              );
            })}
          </div>
          <div style={{display:"flex", justifyContent:"space-between", marginTop:14, fontFamily:"JetBrains Mono", fontSize:10, color:"var(--muted)", letterSpacing:".02em"}}>
            <span>less</span>
            <div style={{display:"flex", gap:3}}>
              {[0,1,2,3].map(n => (
                <span key={n} style={{width:12, height:12, borderRadius:2,
                  background: n === 0 ? "var(--hair)" :
                              n === 1 ? "color-mix(in oklab, var(--accent) 35%, var(--bg))" :
                              n === 2 ? "color-mix(in oklab, var(--accent) 65%, var(--bg))" :
                                         "var(--accent)"
                }}></span>
              ))}
            </div>
            <span>more · 6 posts in Apr</span>
          </div>
        </div>
      </section>

      {/* chronological index */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title">All <em>posts</em></div>
            <div className="count">{all.length} entries · {all.filter(p=>p.draft).length} drafts</div>
          </div>
          <div className="filters">
            <a className="on">all</a><a>published</a><a>drafts</a>
          </div>
        </div>

        {months.map(ym => (
          <div key={ym} className="post-index" style={{marginBottom: 8}}>
            <div className="year">
              {ym.slice(0,4)}<br/>
              <span style={{color:"var(--ink-2)", fontSize:14, fontFamily:"var(--display)", fontStyle:"italic", letterSpacing:"-0.01em"}}>
                {new Date(ym+"-01").toLocaleString("en", {month:"short"})}
              </span>
            </div>
            <div>
              {byMonth[ym].map((p,i) => {
                const Tag = p.draft ? "div" : "a";
                const href = p.draft ? undefined : p.url;
                return (
                  <Tag key={i} className={"post-row" + (p.draft ? " draft" : "")} href={href} data-lang-post={(p.lang||"").toLowerCase()}>
                    <div>
                      <div className="meta-line">
                        <span>{p.date.slice(8)}</span>
                        <span className="d"></span>
                        <span className="cat">{p.cat}</span>
                        <span className="d"></span>
                        <span className="lang">{p.lang}</span>
                        {p.draft && <>
                          <span className="d"></span>
                          <span style={{color:"var(--muted-2)", letterSpacing:".08em"}}>DRAFT</span>
                        </>}
                      </div>
                      <h3 className={"title" + (p.lang==="KO"?" kr":"")}>{p.title}</h3>
                      <p className={"ex" + (p.lang==="KO"?" kr":"")}>{p.ex}</p>
                      <div className="tags-row">
                        {p.tags.map(t => <span key={t} className="tg">{t}</span>)}
                      </div>
                    </div>
                    <div className="right">
                      <span>{p.draft ? "—" : "read"}</span>
                      <span className="arr">{p.draft ? "" : "↗"}</span>
                    </div>
                  </Tag>
                );
              })}
            </div>
          </div>
        ))}
      </section>
    </>
  );
}

export default ArchivesView;
