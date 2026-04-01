from __future__ import annotations

from types import SimpleNamespace

from app.debate.autogen_runner import DebateEngine
from app.llm.models import DebateAgentPositions, DebateAssessmentPayload
from app.llm.openai_client import OpenAIResponsesClient


class FakeAssessmentClient:
    def generate_structured(self, *, schema_model, **_: object):
        return schema_model(
            summary="Os agentes convergiram para um artigo rigoroso.",
            positions=DebateAgentPositions(
                Analitico="Estruturar a progressao argumentativa.",
                Critico="Explicitar limites e riscos.",
                Estrategico="Conectar impacto social e aplicacoes.",
            ),
            needs_more_rounds=False,
            open_questions=[],
        )


def fake_phase_runner(round_number: int, _: str):
    return [
        SimpleNamespace(source="Analitico", content=f"Analitico rodada {round_number}."),
        SimpleNamespace(source="Critico", content=f"Critico rodada {round_number}."),
        SimpleNamespace(source="Estrategico", content=f"Estrategico rodada {round_number}."),
    ]


def test_debate_assessment_schema_uses_explicit_agent_fields():
    schema = OpenAIResponsesClient._build_strict_schema(DebateAssessmentPayload)

    positions_ref = schema["properties"]["positions"]["$ref"]
    assert positions_ref == "#/$defs/DebateAgentPositions"

    nested = schema["$defs"]["DebateAgentPositions"]
    assert nested["required"] == ["Analitico", "Critico", "Estrategico"]
    assert set(nested["properties"].keys()) == {"Analitico", "Critico", "Estrategico"}


def test_debate_engine_preserves_dict_contract_for_positions():
    engine = DebateEngine(
        model="fake-model",
        llm_client=FakeAssessmentClient(),
        phase_runner=fake_phase_runner,
    )

    result = engine.run(
        topic="Crie um artigo sobre inteligencia artificial",
        research_summary="Pesquisa cobrindo fundamentos, riscos e aplicacoes.",
        research_notes=[],
        rounds=3,
    )

    assert result["positions"]["Analitico"] == "Estruturar a progressao argumentativa."
    assert result["positions"]["Critico"] == "Explicitar limites e riscos."
    assert result["positions"]["Estrategico"] == "Conectar impacto social e aplicacoes."
