-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.ai_suggestions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  section_id uuid NOT NULL,
  original_text text,
  suggested_text text NOT NULL,
  instruction text,
  status text NOT NULL DEFAULT 'pending'::text CHECK (status = ANY (ARRAY['pending'::text, 'accepted'::text, 'rejected'::text, 'edited'::text])),
  feedback text,
  created_at timestamp with time zone DEFAULT now(),
  resolved_at timestamp with time zone,
  CONSTRAINT ai_suggestions_pkey PRIMARY KEY (id),
  CONSTRAINT ai_suggestions_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.sections(id)
);
CREATE TABLE public.chat_messages (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL,
  section_id uuid,
  role text NOT NULL CHECK (role = ANY (ARRAY['human'::text, 'ai'::text])),
  content text NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT chat_messages_pkey PRIMARY KEY (id),
  CONSTRAINT chat_messages_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id),
  CONSTRAINT chat_messages_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.sections(id)
);
CREATE TABLE public.document_versions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  section_id uuid NOT NULL,
  suggestion_id uuid,
  content text NOT NULL,
  version_number integer NOT NULL DEFAULT 1,
  author_type text NOT NULL CHECK (author_type = ANY (ARRAY['human'::text, 'ai'::text])),
  is_current boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT document_versions_pkey PRIMARY KEY (id),
  CONSTRAINT document_versions_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.sections(id),
  CONSTRAINT document_versions_suggestion_id_fkey FOREIGN KEY (suggestion_id) REFERENCES public.ai_suggestions(id)
);
CREATE TABLE public.literature_analysis (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL,
  review_content text,
  gaps_content text,
  warnings text,
  status text NOT NULL DEFAULT 'pending'::text CHECK (status = ANY (ARRAY['pending'::text, 'completed'::text, 'reviewed'::text])),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT literature_analysis_pkey PRIMARY KEY (id),
  CONSTRAINT literature_analysis_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id)
);
CREATE TABLE public.literature_citations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  literature_analysis_id uuid NOT NULL,
  source_id uuid NOT NULL,
  formatted_citation text NOT NULL,
  citation_style text,
  position integer NOT NULL DEFAULT 1,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT literature_citations_pkey PRIMARY KEY (id),
  CONSTRAINT literature_citations_literature_analysis_id_fkey FOREIGN KEY (literature_analysis_id) REFERENCES public.literature_analysis(id),
  CONSTRAINT literature_citations_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.sources(id)
);
CREATE TABLE public.project_preferences (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL UNIQUE,
  writing_style text DEFAULT 'formal'::text,
  tone text DEFAULT 'academic'::text,
  target_journal text,
  language text DEFAULT 'English'::text,
  assistance_level text DEFAULT 'moderate'::text CHECK (assistance_level = ANY (ARRAY['light'::text, 'moderate'::text, 'full'::text])),
  citation_style text DEFAULT 'APA'::text,
  grounded_only boolean DEFAULT false,
  llm_provider text DEFAULT 'groq'::text CHECK (llm_provider = ANY (ARRAY['groq'::text, 'gemini'::text, 'anthropic'::text])),
  CONSTRAINT project_preferences_pkey PRIMARY KEY (id),
  CONSTRAINT project_preferences_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id)
);
CREATE TABLE public.projects (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  title text NOT NULL,
  description text,
  status text NOT NULL DEFAULT 'active'::text CHECK (status = ANY (ARRAY['active'::text, 'archived'::text])),
  thread_id text,
  progress integer NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT projects_pkey PRIMARY KEY (id),
  CONSTRAINT projects_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.sections (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL,
  type text NOT NULL CHECK (type = ANY (ARRAY['abstract'::text, 'introduction'::text, 'methodology'::text, 'results'::text, 'discussion'::text, 'conclusion'::text, 'other'::text])),
  title text,
  position integer NOT NULL DEFAULT 1,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT sections_pkey PRIMARY KEY (id),
  CONSTRAINT sections_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id)
);
CREATE TABLE public.sources (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL,
  title text NOT NULL,
  authors text,
  abstract text,
  url text,
  doi text,
  citation_count integer DEFAULT 0,
  relevance_score double precision CHECK (relevance_score >= 0::double precision AND relevance_score <= 1::double precision),
  pdf_url text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT sources_pkey PRIMARY KEY (id),
  CONSTRAINT sources_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id)
);
CREATE TABLE public.users (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  email text NOT NULL UNIQUE,
  full_name text NOT NULL,
  password_hash text NOT NULL,
  academic_position text,
  organization text,
  field_interests text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT users_pkey PRIMARY KEY (id)
);
CREATE TABLE public.visualizations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL,
  section_id uuid,
  type text NOT NULL CHECK (type = ANY (ARRAY['chart'::text, 'table'::text, 'figure'::text])),
  title text,
  raw_data jsonb,
  config jsonb,
  file_path text,
  export_format text DEFAULT 'png'::text CHECK (export_format = ANY (ARRAY['png'::text, 'pdf'::text, 'latex'::text])),
  export_size text,
  color_scheme text DEFAULT 'default'::text,
  details text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT visualizations_pkey PRIMARY KEY (id),
  CONSTRAINT visualizations_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id),
  CONSTRAINT visualizations_section_id_fkey FOREIGN KEY (section_id) REFERENCES public.sections(id)
);