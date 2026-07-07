# Anthropic Academy — Training Recommendations for Data Scientists & Engineers

Anthropic Academy (anthropic.skilljar.com, overview at anthropic.com/learn) is
Anthropic's free, self-paced training platform. Courses are organized into
three tracks — **AI Fluency**, **Product Training**, and **Developer
Deep-Dives** — each ending in a short quiz and an official completion
certificate. No subscription or API key is required to enroll, though the
developer-track courses assume access to a Claude API key for hands-on
exercises.

This doc maps the current catalog to two audiences on our team — **Data
Scientists / ML Engineers** and **Software / Platform Engineers** — and calls
out which real, on-the-job skills each course unlocks. Where relevant, it
also notes how a course connects to work we're already doing (e.g. the
document-classification pipeline in this repo).

> Catalog note: Anthropic Academy is actively growing (it launched in
> March 2026 with ~14 courses and has since expanded past 17). Course names
> and availability should be double-checked at anthropic.skilljar.com before
> finalizing a cohort schedule.

---

## Track 1 — Data Scientists / ML Engineers

| Order | Course | Track | Skills achieved |
|---|---|---|---|
| 1 | **Claude Platform 101** | Product Training | Navigating Claude.ai / Console, Projects, and Artifacts; basic conversational prompting; when to reach for Claude vs. a custom model. |
| 2 | **AI Fluency: Framework and Foundations** | AI Fluency | The "4E" framework (Effective, Efficient, Ethical, Safe) for evaluating and delegating work to an LLM; judging output quality and knowing when *not* to trust a model. |
| 3 | **Building with the Claude API** | Developer Deep-Dives | Anthropic API fundamentals; advanced prompting; tool use / function calling; retrieval-augmented generation (RAG); multimodal input (text, images, PDFs); architecting agentic workflows (chaining, routing, parallelization). |
| 4 | **Introduction to MCP** | Developer Deep-Dives | Model Context Protocol architecture; building MCP servers/clients in Python; exposing internal data sources, feature stores, or notebooks as tools Claude can call. |
| 5 | **Claude Code 101** | Product Training | Using Claude Code as a terminal-based coding/analysis assistant; the Explore → Plan → Code → Commit loop for scoping data-pipeline changes. |
| 6 (optional) | **Introduction to Agent Skills** | Developer Deep-Dives | Packaging a repeatable analysis or preprocessing routine as a reusable `SKILL.md` that Claude applies automatically. |

**Why this matters for this repo:** the classifier pipeline here
(`src/classifier/preprocessor.py`, `features.py`, `models.py`) hand-rolls
TF-IDF and embedding-based classification. "Building with the Claude API"
and "Introduction to MCP" directly translate into being able to (a)
prototype a Claude-based zero/few-shot classifier as a baseline comparison
against the TF-IDF/embedding models in `Evaluator.compare()`, and (b) expose
`SyntheticDataset`/`TextPreprocessor` as MCP tools so Claude can drive
end-to-end labeling or data-quality checks.

## Track 2 — Software / Platform Engineers

| Order | Course | Track | Skills achieved |
|---|---|---|---|
| 1 | **Claude Code 101** | Product Training | Installing and configuring Claude Code; core agentic workflow; scoping tasks correctly. |
| 2 | **Claude Code in Action** | Developer Deep-Dives | Context management at scale; GitHub integration; wiring MCP servers into the dev loop; `CLAUDE.md`/hooks/subagent customization; production best practices. |
| 3 | **Building with the Claude API** | Developer Deep-Dives | Full API surface (tool use, streaming, multimodal); agent architecture patterns (chaining, routing, parallelization); deploying Claude Code and Computer Use for automation. |
| 4 | **Introduction to MCP → MCP: Advanced Topics** | Developer Deep-Dives | Building MCP servers/clients from scratch; production concerns — sampling, notifications, transport (stdio vs. Streamable HTTP). |
| 5 | **Introduction to Agent Skills + Subagents & Skills Management** | Developer Deep-Dives | Authoring, versioning, and distributing Skills/subagents across a team; troubleshooting Skill discovery and invocation. |
| 6 | **AI Fluency: Framework and Foundations** | AI Fluency | Shared vocabulary with the DS/product side of the org for evaluating AI-assisted output responsibly. |

**Follow-on for advanced engineers:** the **Claude Certified Architect**
exam (anthropiccertifications.com) builds on the Developer Deep-Dives track
— agentic loops, multi-agent orchestration, MCP tool design, and context
engineering — and is worth flagging to engineers who complete the full
developer track and want a formal credential.

---

## Suggested Sequencing

1. **Week 1 (shared foundation, all attendees):** Claude Platform 101 +
   AI Fluency: Framework and Foundations. Establishes common vocabulary and
   responsible-use norms before anyone touches the API.
2. **Week 2 (split by role):** DS/ML track starts Building with the Claude
   API; Engineering track starts Claude Code 101 → Claude Code in Action.
3. **Week 3:** Both tracks converge on Introduction to MCP, since it's
   relevant to both connecting data tools (DS) and production tool servers
   (Engineering).
4. **Week 4 (optional, for teams building reusable internal tooling):**
   Introduction to Agent Skills, aimed at whoever owns shared
   CLAUDE.md/skills conventions for the team.

## Sources

- [Anthropic Courses (Skilljar catalog)](https://anthropic.skilljar.com/)
- [AI Learning Resources & Guides from Anthropic](https://www.anthropic.com/learn)
- [Building with the Claude API](https://anthropic.skilljar.com/claude-with-the-anthropic-api)
- [Claude Platform 101](https://anthropic.skilljar.com/claude-platform-101)
- [Claude Certified Architect Courses](https://www.anthropiccertifications.com/courses)
- [Anthropic Academy Guide — Termdock](https://www.termdock.com/en/blog/anthropic-academy-claude-courses-guide)
