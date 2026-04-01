"""Focused section recovery orchestration."""

from __future__ import annotations

from app.debate.autogen_runner import DebateEngine
from app.llm.models import ResearchNote, SectionRecoveryPlan, merge_prompt_pack
from app.llm.openai_client import OpenAIResponsesClient
from app.prompts.builders import build_section_recovery_prompt
from app.research.parser import normalize_sources
from app.research.summarizer import ResearchSummarizer
from app.research.web_search import SearchBackend
from app.workflow.error_policy import WorkflowErrorPolicy
from app.workflow.models import SectionState, WorkflowStateEnvelope


class SectionRecoveryService:
    """Prepare focused recovery inputs and run the section mini-debate."""

    def __init__(
        self,
        *,
        llm_client: OpenAIResponsesClient,
        search_backend: SearchBackend,
        research_summarizer: ResearchSummarizer,
        debate_engine: DebateEngine,
        error_policy: WorkflowErrorPolicy,
        research_max_sources: int,
        article_profile: str,
    ) -> None:
        self._llm_client = llm_client
        self._search_backend = search_backend
        self._research_summarizer = research_summarizer
        self._debate_engine = debate_engine
        self._error_policy = error_policy
        self._research_max_sources = research_max_sources
        self._article_profile = article_profile

    def prepare_recovery(
        self,
        *,
        envelope: WorkflowStateEnvelope,
        section: SectionState,
        node_name: str,
    ) -> tuple[WorkflowStateEnvelope, SectionState]:
        prompt_envelope = build_section_recovery_prompt(
            normalized_request=envelope.normalized_request,
            article_goal=envelope.article_goal,
            knowledge_domain=envelope.knowledge_domain,
            disciplinary_lens=envelope.disciplinary_lens,
            section_heading=section.heading,
            section_kind=section.kind,
            section_purpose=section.purpose,
            section_bullets=section.bullets,
            research_summary=envelope.research.summary,
            debate_summary=envelope.debate.summary,
            review_summary=section.review_summary,
            weaknesses=section.weaknesses,
            revision_requirements=section.revision_requirements,
            prompt_improvements=section.prompt_improvements,
            draft_md=section.draft_md,
        )
        try:
            plan = self._llm_client.generate_structured(
                envelope=prompt_envelope,
                schema_model=SectionRecoveryPlan,
                temperature=0.2,
            )
        except Exception as exc:
            envelope = self._error_policy.record_fallback(
                envelope,
                node=node_name,
                exc=exc,
                decision="fallback:section_recovery_plan",
            )
            plan = self._fallback_plan(envelope, section)

        max_results = max(2, min(4, self._research_max_sources))
        results = [
            self._search_backend.search(query, max_results=max_results)
            for query in plan.research_queries
        ]
        references = normalize_sources(results, max_results=self._research_max_sources)
        summary_payload = self._research_summarizer.summarize(
            topic=f"{envelope.normalized_request} - secao {section.heading}",
            knowledge_domain=envelope.knowledge_domain,
            disciplinary_lens=envelope.disciplinary_lens,
            evidence_requirements=envelope.evidence_requirements,
            results=results,
            references=references,
            minimum_words=max(section.target_words, 1200),
            article_profile=self._article_profile,
        )
        resolved_pack = merge_prompt_pack(envelope.debate.prompt_pack, plan.prompt_pack_override)

        section.recovery_problem_summary = plan.problem_summary
        section.section_research_queries = plan.research_queries
        section.section_research_notes = summary_payload.notes
        section.section_research_summary = summary_payload.research_summary
        section.section_debate_prompt = plan.debate_prompt
        section.prompt_pack_override = plan.prompt_pack_override
        section.resolved_prompt_pack = resolved_pack
        section.section_debate_summary = ""
        section.section_agent_positions = envelope.debate.positions.__class__()
        return envelope, section

    def run_section_debate(
        self,
        *,
        envelope: WorkflowStateEnvelope,
        section: SectionState,
    ) -> tuple[WorkflowStateEnvelope, SectionState]:
        debate_result = self._debate_engine.run(
            topic=section.section_debate_prompt,
            research_summary=section.section_research_summary,
            research_notes=[
                note.model_dump() if isinstance(note, ResearchNote) else note
                for note in section.section_research_notes
            ],
            rounds=3,
            prior_transcript=None,
            start_round=1,
            open_questions=section.revision_requirements or section.weaknesses,
            shared_context=section.resolved_prompt_pack.shared_context,
            focus_axes=section.resolved_prompt_pack.focus_axes,
            domain_terms=section.resolved_prompt_pack.domain_terms,
            agent_specializations=section.resolved_prompt_pack.agent_map(),
        )
        section.section_debate_summary = str(debate_result["summary"])
        section.section_agent_positions = envelope.debate.positions.from_mapping(debate_result["positions"])
        return envelope, section

    def _fallback_plan(
        self,
        envelope: WorkflowStateEnvelope,
        section: SectionState,
    ) -> SectionRecoveryPlan:
        topic = envelope.normalized_request
        heading = section.heading
        return SectionRecoveryPlan(
            problem_summary=section.review_summary or "A secao precisa de evidencia e enquadramento mais preciso.",
            research_queries=[
                f"{topic} {heading} revisao sistematica",
                f"{topic} {heading} evidencias recentes",
                f"{topic} {heading} controversias e limitacoes",
            ][:3],
            debate_prompt=(
                f"Refine a secao '{heading}' do artigo sobre '{topic}', cobrindo as lacunas "
                "identificadas na revisao e propondo a melhor formulacao editorial."
            ),
            prompt_pack_override={
                "shared_context": f"Recuperacao focada da secao {heading} no contexto do tema {topic}.",
                "focus_axes": section.revision_requirements[:3] or section.weaknesses[:3],
                "domain_terms": section.bullets[:3],
                "Analitico": "Reestruture a secao com logica mais rigorosa e encadeamento claro.",
                "Critico": "Aponte extrapolacoes, lacunas de evidencia e simplificacoes indevidas.",
                "Estrategico": "Conecte a secao ao argumento global e a implicacoes mais amplas do tema.",
            },
        )
