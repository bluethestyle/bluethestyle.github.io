import React from "react";
import { SITE, CURRENT_WORK, COAUTHORS, SERIES } from "../data.js";

// ========= HOME TAB (faithful to bluethestyle.github.io structure) =========

function PLEDiagram() {
  // schematic of the 13-task PLE, 7 experts → distilled to LightGBM
  const experts = 7;
  const tasks = 13;
  return (
    <svg viewBox="0 0 300 220" preserveAspectRatio="xMidYMid meet" className="viz-svg">
      <defs>
        <pattern id="grid-work" width="16" height="16" patternUnits="userSpaceOnUse">
          <path d="M 16 0 L 0 0 0 16" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.15"/>
        </pattern>
      </defs>
      <rect width="300" height="220" fill="url(#grid-work)" color="var(--muted-2)"/>

      {/* 7 experts column (left) */}
      {Array.from({length: experts}).map((_, i) => {
        const y = 22 + i * 26;
        return (
          <g key={"e"+i}>
            <rect x="18" y={y-6} width="54" height="12" rx="2" fill="var(--surface)" stroke="var(--hair)"/>
            <text x="45" y={y+3} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="7" fill="var(--muted)">E{i+1}</text>
          </g>
        );
      })}
      {/* gate (middle) */}
      <rect x="110" y="78" width="54" height="64" rx="4" fill="var(--accent-soft)" stroke="var(--accent)"/>
      <text x="137" y="102" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="8" fill="var(--accent-ink)" fontWeight="500">GATE</text>
      <text x="137" y="115" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="7" fill="var(--accent-ink)">PLE</text>
      <text x="137" y="128" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="6.5" fill="var(--accent-ink)" opacity="0.7">×7</text>

      {/* connections experts → gate */}
      {Array.from({length: experts}).map((_, i) => {
        const y1 = 22 + i * 26;
        return <line key={"l"+i} x1="72" y1={y1} x2="110" y2="110" stroke="var(--accent)" strokeWidth="0.8" opacity="0.55"/>;
      })}

      {/* 13 task heads (right) */}
      {Array.from({length: tasks}).map((_, i) => {
        const row = i % 7;
        const col = Math.floor(i / 7);
        const y = 22 + row * 26;
        const x = 196 + col * 48;
        return (
          <g key={"t"+i}>
            <rect x={x} y={y-5} width="40" height="10" rx="2" fill="var(--bg-2)" stroke="var(--hair)"/>
            <text x={x+20} y={y+3} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="6.5" fill="var(--muted)">T{String(i+1).padStart(2,"0")}</text>
            <line x1="164" y1="110" x2={x} y2={y} stroke="var(--muted-2)" strokeWidth="0.5" strokeDasharray="1.5 2" opacity="0.6"/>
          </g>
        );
      })}

      {/* labels */}
      <text x="18" y="12" fontFamily="JetBrains Mono" fontSize="7" fill="var(--muted)" letterSpacing="0.5">EXPERTS · 7</text>
      <text x="110" y="72" fontFamily="JetBrains Mono" fontSize="7" fill="var(--muted)" letterSpacing="0.5">PLE GATE</text>
      <text x="196" y="12" fontFamily="JetBrains Mono" fontSize="7" fill="var(--muted)" letterSpacing="0.5">TASKS · 13</text>

      {/* distillation line */}
      <path d="M 137 150 C 137 180, 150 198, 200 202" stroke="var(--accent)" strokeWidth="1.4" fill="none"/>
      <rect x="200" y="196" width="80" height="14" rx="2" fill="var(--ink)" />
      <text x="240" y="206" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="7.5" fill="var(--bg)" letterSpacing="0.5">→ LIGHTGBM / λ</text>
    </svg>
  );
}

function HomeView({ lang, posts = [] }) {
  const POSTS = posts;
  const KO = lang === "ko";
  const latest = POSTS.filter(p => !p.draft).slice(0, 4);

  return (
    <>
      {/* HERO */}
      <section className="hero">
        <div className="label">
          <span className="pulse"></span>
          Working notes · est. 2026 · 기록과 정리
        </div>
        <h1>
          Seonkyu <em>Jeong</em> — notes on <span className="hl">financial&nbsp;AI</span>, model&nbsp;risk, and the slow integration of <em>agents</em> into regulated workflows.
        </h1>
        <p>
          <b>Independent researcher · Seoul.</b> {SITE.bio}
        </p>
        <p>{SITE.bio2}</p>
        <div>
          <span className="frm">◆ GARP FRM · Financial Risk Manager</span>
        </div>
      </section>

      {/* CURRENT WORK */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title">Current <em>work</em></div>
            <div className="count">2026 · active</div>
          </div>
        </div>

        <div className="work-card">
          <div className="work-main">
            <div className="work-tag"><span className="d"></span>Recommendation · MRM · Agentic</div>
            <h3>{CURRENT_WORK.title}</h3>
            <p className="d">{CURRENT_WORK.desc}</p>
            <p className="m">{CURRENT_WORK.meta}</p>

            <div className="work-links">
              {CURRENT_WORK.links.map((l,i) => (
                <a key={i} href={l.href}>
                  <span className="lt">{l.tag}</span>
                  <span>{l.label}<br/><span style={{fontFamily:"JetBrains Mono",fontSize:10,color:"var(--muted-2)",letterSpacing:".02em"}}>{l.href.replace("https://","")}</span></span>
                  <span className="arr">↗</span>
                </a>
              ))}
            </div>
          </div>

          <div className="work-viz">
            <div className="viz-top"><span>fig. 01</span><b>hetero-expert PLE</b></div>
            <PLEDiagram />
            <div className="viz-nums">
              <div><div className="n">7</div><div className="l">experts</div></div>
              <div><div className="n">13</div><div className="l">tasks</div></div>
              <div><div className="n">λ</div><div className="l">lambda</div></div>
            </div>
          </div>
        </div>
      </section>

      {/* CO-AUTHORS */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title">Co-<em>authors</em></div>
            <div className="count">3 · personal time</div>
          </div>
        </div>
        <div className="authors">
          {COAUTHORS.map((a,i) => (
            <div key={i} className={"author" + (a.lead ? " lead" : "")}>
              <div className="init">{a.name.split(" ").map(w => w[0]).join("")}</div>
              <div className="nm">{a.name}</div>
              <div className="rl">{a.role}</div>
            </div>
          ))}
        </div>
      </section>

      {/* SERIES */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title"><em>Series</em></div>
            <div className="count">EN · KO parallel</div>
          </div>
        </div>
        <div className="series-grid">
          {SERIES.map(s => (
            <div key={s.slug} className="series">
              <div className="top">
                <span className="cat">[{s.tag}]</span>
                <span className="prog">ep {s.ep} of {s.total}</span>
              </div>
              <h4>{s.title}</h4>
              <p>{s.desc}</p>
              <div className="bar"><span style={{width: (s.ep/s.total*100) + "%"}}></span></div>
              <div className="lang">
                <a>read · english ↗</a>
                <span className="sep">·</span>
                <a>한국어로 읽기 ↗</a>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* LATEST POSTS */}
      <section className="block">
        <div className="sec-head">
          <div className="sec-head-l">
            <div className="title">Latest <em>posts</em></div>
            <div className="count">{latest.length} · 전체 {POSTS.filter(p=>!p.draft).length}</div>
          </div>
          <div className="filters">
            <a className="on">all</a><a>en</a><a>한국어</a>
          </div>
        </div>

        <div className="post-index">
          <div className="year">2026</div>
          <div>
            {latest.map((p,i) => (
              <a key={i} className="post-row" href={p.url || "#"} data-lang-post={(p.lang||"").toLowerCase()}>
                <div>
                  <div className="meta-line">
                    <span>{p.date.slice(5).replace("-"," · ")}</span>
                    <span className="d"></span>
                    <span className="cat">{p.cat}</span>
                    <span className="d"></span>
                    <span className="lang">{p.lang}</span>
                  </div>
                  <h3 className={"title" + (p.lang==="KO"?" kr":"")}>{p.title}</h3>
                  <p className={"ex" + (p.lang==="KO"?" kr":"")}>{p.ex}</p>
                  <div className="tags-row">
                    {p.tags.map(t => <span key={t} className="tg">{t}</span>)}
                  </div>
                </div>
                <div className="right">
                  <span>read</span>
                  <span className="arr">↗</span>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Contact strip */}
        <div className="contact-bar">
          <span><span className="k">contact</span> &nbsp; <span className="v">{SITE.contact}</span></span>
          <span className="sep">·</span>
          <span><span className="k">orcid</span> &nbsp; <a>{SITE.orcid} ↗</a></span>
          <span className="sep">·</span>
          <span><span className="k">github</span> &nbsp; <a>@{SITE.github} ↗</a></span>
          <span style={{marginLeft:"auto"}}><span className="k">license</span> &nbsp; <span className="v">content CC BY-NC 4.0 · code MIT</span></span>
        </div>
      </section>
    </>
  );
}

export default HomeView;
