from __future__ import annotations

from app.debate.agents import DEBATE_AGENTS
from app.debate.prompts import build_agent_system_message, build_phase_task
from app.llm.models import IntakePlan, ResearchNote
from app.llm.openai_client import OpenAIResponsesClient


def test_intake_plan_schema_exposes_explicit_debate_prompt_pack():
    schema = OpenAIResponsesClient._build_strict_schema(IntakePlan)

    pack_ref = schema["properties"]["debate_prompt_pack"]["$ref"]
    assert pack_ref == "#/$defs/DebatePromptPack"

    pack_schema = schema["$defs"]["DebatePromptPack"]
    assert set(pack_schema["properties"].keys()) == {
        "shared_context",
        "focus_axes",
        "domain_terms",
        "Analitico",
        "Critico",
        "Estrategico",
    }
    assert pack_schema["required"] == [
        "shared_context",
        "focus_axes",
        "domain_terms",
        "Analitico",
        "Critico",
        "Estrategico",
    ]


def test_agent_prompt_receives_topic_specialization():
    prompt = build_agent_system_message(
        DEBATE_AGENTS[0],
        shared_context="Tema de meio ambiente com foco em politica climatica e biodiversidade.",
        focus_axes=["governanca ambiental", "trade-offs regulatorios"],
        domain_terms=["descarbonizacao", "licenciamento ambiental", "justica climatica"],
        agent_specialization="Estruture o debate por causalidade, escala temporal e atores institucionais.",
    )

    assert "Contexto especializado do tema" in prompt
    assert "governanca ambiental" in prompt
    assert "justica climatica" in prompt
    assert "atores institucionais" in prompt


def test_phase_task_includes_shared_context_focus_axes_and_terms():
    task = build_phase_task(
        topic="Debater impactos de politicas ambientais costeiras.",
        research_summary="Resumo da pesquisa sobre erosao costeira e governanca local.",
        research_notes=[
            ResearchNote(
                title="Fonte ambiental",
                source="Instituto Costeiro",
                url="https://example.com/costa",
                summary="Discute erosao, ocupacao urbana e adaptacao climatica.",
            )
        ],
        round_number=1,
        shared_context="Tema ambiental com forte interface entre clima, territorio e politicas publicas.",
        focus_axes=["adaptacao costeira", "governanca multinivel"],
        domain_terms=["erosao costeira", "resiliencia urbana"],
    )

    assert "Contexto especializado do tema" in task
    assert "adaptacao costeira" in task
    assert "resiliencia urbana" in task
    assert "Fonte ambiental" in task
