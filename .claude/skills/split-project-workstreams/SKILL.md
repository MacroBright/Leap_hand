---
name: split-project-workstreams
description: Use when starting a complex multi-module project and wanting to split work into parallel Claude Code sessions, or when a project has grown too large for a single conversation window and needs structured decomposition into independent workstreams
---

# Split Project Into Parallel Workstreams

## Overview

Analyze a project's codebase and split it into independent parallel workstreams — each with its own briefing file — so multiple Claude Code sessions can work concurrently without stepping on each other.

**Core principle:** Ask before you act. Present options, get confirmation, then create files.

## When to Use

- Starting a new multi-module project and want parallel development from day one
- Existing project has grown too large for one conversation window
- Project spans multiple technology stacks (e.g., Python + C + ML + docs)
- User says "split into parallel windows", "multiple chat sessions", "workstreams", or "并行窗口"

**When NOT to use:**
- Single-module projects (one language, <10 files)
- The user explicitly wants everything in one window
- Quick one-off tasks with no ongoing development

## Process

```
Explore → Categorize → Present Options → Confirm → Create Files
```

### Phase 1: Explore the Codebase

Read the project's CLAUDE.md (if exists), scan top-level directories, identify:
- Technology stacks present (Python, TypeScript, C, etc.)
- Natural module boundaries (separate directories with different concerns)
- Existing dependency relationships

### Phase 2: Categorize — Present 3 Options

**REQUIRED:** You MUST present these 3 approaches and ask the user to choose. Do NOT assume one.

| Approach | Logic | Best for |
|----------|-------|----------|
| **A. By Technology Layer** | Split by language/framework/module boundary | Projects with clear tech stack separations (backend/frontend/firmware/ML) |
| **B. By Workflow Stage** | Split by development phase (dev → review → deploy) | Projects where quality gates matter more than parallel speed |
| **C. By Sub-project** | Split by deliverable/feature area | Feature-heavy projects with independent user-facing modules |

Ask: *"Which split approach fits your workflow? A (tech layers), B (workflow stages), or C (sub-projects)?"*

### Phase 3: Propose a Split

Based on the chosen approach, propose 3-5 workstreams. **Hard cap: 5 workstreams** for projects under 100 files. Only exceed for very large projects (200+ files, 4+ technology stacks).

Each workstream proposal must include:
- Name + emoji (for visual identification)
- Which files/modules it covers
- One-sentence responsibility statement

Present to user and ask: *"Does this split look right? Any to merge or split further?"*

### Phase 4: Create Files (Only After Confirmation)

**CRITICAL: Wait for explicit user confirmation before creating ANY files.**

Create these files:

1. **`.claude/workstreams/0X-name.md`** — one per workstream (keep each under 80 lines)
2. **Update `CLAUDE.md`** — add a concise "Multi-Window Workflow" section (under 15 lines)
3. **Create memory entry** — in `memory/multi-window-workflow.md`

## Briefing File Template

Each briefing file MUST follow this structure. Keep it lean — the workstream session itself will fill in details.

```markdown
# Window N: [Name] ([Emoji])

## Scope
| File/Module | Purpose |
|-------------|---------|
| `path/to/file` | What it does |

## Current State
- [ ] Task 1
- [ ] Task 2

## Next Tasks (prioritized)
1. 🔴 Most urgent
2. 🟡 Nice to have
3. 🟢 Future

## Interfaces
### Output → Window X
- What this window produces for others
### Input ← Window Y
- What this window consumes from others

## Session Start
Load `.claude/workstreams/0X-name.md`. Current task: [describe]

## References
- CLAUDE.md
- memory/handoff-*.md
```

**Anti-patterns to avoid in briefs:**
- ❌ Detailed implementation plans (belongs in the workstream session itself)
- ❌ Full API specs or protocol docs (reference external files instead)
- ❌ 200+ line briefs (split into sub-files if needed)
- ❌ Duplicating CLAUDE.md content

## CLAUDE.md Update Template

Add one concise section. Do NOT add ASCII art diagrams, launch orders, or communication protocols.

```markdown
## N. Multi-Window Parallel Workflow

This project is split into N independent Claude Code sessions:

| # | Focus | Brief | Covers |
|---|-------|-------|--------|
| 1 | Name | `.claude/workstreams/01-name.md` | Key files |

**Usage:** New Claude Code window → `Load .claude/workstreams/0X-name.md, task: [describe]`

**Dependencies:** W1 → W2 (data), W3 ↔ W1 (protocols), W4 → all (docs).
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Creating briefs without user confirmation | Present the split proposal FIRST, wait for approval |
| Over-splitting (6+ workstreams for small projects) | Cap at 5 for projects under 100 files |
| Writing 200+ line briefing files | Target 60-80 lines. The workstream session fills in details |
| Skipping the 3-approach question | Always ask A/B/C before proposing |
| Adding ASCII architecture diagrams to CLAUDE.md | Keep it to a simple table + one-line dependency description |
| Not documenting inter-workstream interfaces | Each brief MUST have Input/Output section |

## Red Flags — STOP and Ask the User

- "I can just pick the right split approach myself" → The user must choose, not you
- "This project is simple, I'll just create 2 workstreams" → Still ask A/B/C
- "I'll write detailed briefs upfront to save time" → Minimal briefs. Details go in the session
- "6 workstreams feels right for this" → Cap at 5 unless project is very large (200+ files)
