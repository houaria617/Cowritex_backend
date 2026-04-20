--Raw SQL — all app tables definitions
-- projects, documents, edit_history, sources, hitl_events
-- ============================================================
-- CoWriteX — Supabase PostgreSQL Schema
-- ============================================================

-- Enable UUID generation
create extension if not exists "pgcrypto";

-- ============================================================
-- 1. users
-- ============================================================
create table if not exists users (
    id                  uuid primary key default gen_random_uuid(),
    email               text unique not null,
    full_name           text not null,
    password_hash       text not null,
    academic_position   text,
    organization        text,
    field_interests     text,
    created_at          timestamptz default now(),
    updated_at          timestamptz default now()
);

-- ============================================================
-- 2. projects
-- ============================================================
create table if not exists projects (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references users(id) on delete cascade,
    title       text not null,
    description text,
    status      text not null default 'active' check (status in ('active', 'archived')),
    thread_id   text,                      -- LangGraph checkpointer thread ID
    progress    int  not null default 0 check (progress between 0 and 100),
    created_at  timestamptz default now(),
    updated_at  timestamptz default now()
);

-- ============================================================
-- 3. project_preferences  (one-to-one with projects)
-- ============================================================
create table if not exists project_preferences (
    id               uuid primary key default gen_random_uuid(),
    project_id       uuid unique not null references projects(id) on delete cascade,
    writing_style    text default 'formal',
    tone             text default 'academic',
    target_journal   text,
    language         text default 'English',
    assistance_level text default 'moderate' check (assistance_level in ('light', 'moderate', 'full')),
    citation_style   text default 'APA',
    grounded_only    boolean default false,
    llm_provider     text default 'groq' check (llm_provider in ('groq', 'gemini', 'anthropic'))
);

-- ============================================================
-- 4. sections
-- ============================================================
create table if not exists sections (
    id          uuid primary key default gen_random_uuid(),
    project_id  uuid not null references projects(id) on delete cascade,
    type        text not null check (type in ('abstract','introduction','methodology','results','discussion','conclusion','other')),
    title       text,
    position    int  not null default 1,
    created_at  timestamptz default now(),
    updated_at  timestamptz default now()
);

-- ============================================================
-- 5. ai_suggestions
-- ============================================================
create table if not exists ai_suggestions (
    id              uuid primary key default gen_random_uuid(),
    section_id      uuid not null references sections(id) on delete cascade,
    original_text   text,
    suggested_text  text not null,
    instruction     text,
    status          text not null default 'pending' check (status in ('pending','accepted','rejected','edited')),
    feedback        text,
    created_at      timestamptz default now(),
    resolved_at     timestamptz
);

-- ============================================================
-- 6. document_versions
-- ============================================================
create table if not exists document_versions (
    id              uuid primary key default gen_random_uuid(),
    section_id      uuid not null references sections(id) on delete cascade,
    suggestion_id   uuid references ai_suggestions(id) on delete set null,
    content         text not null,
    version_number  int  not null default 1,
    author_type     text not null check (author_type in ('human','ai')),
    is_current      boolean not null default true,
    created_at      timestamptz default now()
);

-- Only one current version per section
create unique index if not exists one_current_version_per_section
    on document_versions (section_id)
    where is_current = true;

-- ============================================================
-- 7. chat_messages
-- ============================================================
create table if not exists chat_messages (
    id          uuid primary key default gen_random_uuid(),
    project_id  uuid not null references projects(id) on delete cascade,
    section_id  uuid references sections(id) on delete set null,
    role        text not null check (role in ('human','ai')),
    content     text not null,
    created_at  timestamptz default now()
);

-- ============================================================
-- 8. sources
-- ============================================================
create table if not exists sources (
    id              uuid primary key default gen_random_uuid(),
    project_id      uuid not null references projects(id) on delete cascade,
    title           text not null,
    authors         text,
    abstract        text,
    url             text,
    doi             text,
    citation_count  int  default 0,
    relevance_score float check (relevance_score between 0 and 1),
    pdf_url         text,
    created_at      timestamptz default now()
);

-- ============================================================
-- 9. literature_analysis
-- ============================================================
create table if not exists literature_analysis (
    id              uuid primary key default gen_random_uuid(),
    project_id      uuid not null references projects(id) on delete cascade,
    review_content  text,
    gaps_content    text,
    warnings        text,
    status          text not null default 'pending' check (status in ('pending','completed','reviewed')),
    created_at      timestamptz default now(),
    updated_at      timestamptz default now()
);

-- ============================================================
-- 10. literature_citations
-- ============================================================
create table if not exists literature_citations (
    id                      uuid primary key default gen_random_uuid(),
    literature_analysis_id  uuid not null references literature_analysis(id) on delete cascade,
    source_id               uuid not null references sources(id) on delete cascade,
    formatted_citation      text not null,
    citation_style          text,
    position                int  not null default 1,
    created_at              timestamptz default now()
);

-- ============================================================
-- 11. visualizations
-- ============================================================
create table if not exists visualizations (
    id              uuid primary key default gen_random_uuid(),
    project_id      uuid not null references projects(id) on delete cascade,
    section_id      uuid references sections(id) on delete set null,
    type            text not null check (type in ('chart','table','figure')),
    title           text,
    raw_data        jsonb,
    config          jsonb,
    file_path       text,
    export_format   text default 'png' check (export_format in ('png','pdf','latex')),
    export_size     text,
    color_scheme    text default 'default',
    details         text,
    created_at      timestamptz default now(),
    updated_at      timestamptz default now()
);

-- ============================================================
-- updated_at triggers
-- ============================================================
create or replace function update_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create trigger trg_users_updated_at
    before update on users
    for each row execute function update_updated_at();

create trigger trg_projects_updated_at
    before update on projects
    for each row execute function update_updated_at();

create trigger trg_sections_updated_at
    before update on sections
    for each row execute function update_updated_at();

create trigger trg_literature_analysis_updated_at
    before update on literature_analysis
    for each row execute function update_updated_at();

create trigger trg_visualizations_updated_at
    before update on visualizations
    for each row execute function update_updated_at();

-- ============================================================
-- Row Level Security (enable for Supabase)
-- ============================================================
alter table users                enable row level security;
alter table projects             enable row level security;
alter table project_preferences  enable row level security;
alter table sections             enable row level security;
alter table ai_suggestions       enable row level security;
alter table document_versions    enable row level security;
alter table chat_messages        enable row level security;
alter table sources              enable row level security;
alter table literature_analysis  enable row level security;
alter table literature_citations enable row level security;
alter table visualizations       enable row level security;

-- Basic RLS: users can only touch their own data
-- (expand as needed — these are starter policies)

create policy "users_own_row" on users
    using (auth.uid()::text = id::text);

create policy "projects_owner" on projects
    using (user_id = auth.uid());

create policy "project_preferences_via_project" on project_preferences
    using (project_id in (select id from projects where user_id = auth.uid()));

create policy "sections_via_project" on sections
    using (project_id in (select id from projects where user_id = auth.uid()));

create policy "ai_suggestions_via_section" on ai_suggestions
    using (section_id in (
        select s.id from sections s
        join projects p on p.id = s.project_id
        where p.user_id = auth.uid()
    ));

create policy "document_versions_via_section" on document_versions
    using (section_id in (
        select s.id from sections s
        join projects p on p.id = s.project_id
        where p.user_id = auth.uid()
    ));

create policy "chat_messages_via_project" on chat_messages
    using (project_id in (select id from projects where user_id = auth.uid()));

create policy "sources_via_project" on sources
    using (project_id in (select id from projects where user_id = auth.uid()));

create policy "literature_analysis_via_project" on literature_analysis
    using (project_id in (select id from projects where user_id = auth.uid()));

create policy "literature_citations_via_analysis" on literature_citations
    using (literature_analysis_id in (
        select la.id from literature_analysis la
        join projects p on p.id = la.project_id
        where p.user_id = auth.uid()
    ));

create policy "visualizations_via_project" on visualizations
    using (project_id in (select id from projects where user_id = auth.uid()));