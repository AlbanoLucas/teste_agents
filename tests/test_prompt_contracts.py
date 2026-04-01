from app.llm.models import DebateAssessmentPayload, DebatePromptPackOverride, ResolvedDebatePromptPack, merge_prompt_pack
from app.llm.openai_client import OpenAIResponsesClient
from app.prompts.fragments import join_prompt_blocks, render_approved_sections_block, render_list_block, render_references_block
from app.llm.models import SourceReference


def test_merge_prompt_pack_preserves_global_values_when_override_is_blank() -> None:
    global_pack = ResolvedDebatePromptPack(
        shared_context="Contexto global do tema.",
        focus_axes=["causas", "impactos", "respostas"],
        domain_terms=["mitigacao", "adaptacao"],
        Analitico="Organize o argumento com rigor.",
        Critico="Tensione as evidencias.",
        Estrategico="Conecte o tema a impactos amplos.",
    )
    override = DebatePromptPackOverride(
        shared_context="   ",
        focus_axes=[],
        domain_terms=None,
        Analitico="",
        Critico=None,
        Estrategico="   ",
    )

    merged = merge_prompt_pack(global_pack, override)

    assert merged == global_pack


def test_debate_assessment_schema_uses_explicit_fixed_agent_positions() -> None:
    schema = OpenAIResponsesClient._build_strict_schema(DebateAssessmentPayload)

    assert set(schema["required"]) == {"summary", "positions", "needs_more_rounds", "open_questions"}
    positions_ref = schema["properties"]["positions"]["$ref"]
    positions_schema = schema["$defs"][positions_ref.split("/")[-1]]
    assert positions_schema["type"] == "object"
    assert positions_schema["additionalProperties"] is False
    assert set(positions_schema["required"]) == {"Analitico", "Critico", "Estrategico"}


def test_prompt_fragments_omit_empty_blocks_and_apply_limits() -> None:
    prompt = join_prompt_blocks(
        render_list_block("Lista vazia", []),
        render_approved_sections_block(
            ["Secao 1: texto", "Secao 2: texto", "Secao 3: texto", "Secao 4: texto"],
            limit=3,
        ),
        render_references_block(
            "Referencias",
            [
                SourceReference(title=f"Fonte {index}", source="Org", url=f"https://example.com/{index}")
                for index in range(10)
            ],
            limit=8,
        ),
    )

    assert "Lista vazia" not in prompt
    assert prompt.count("- Secao") == 3
    assert prompt.count("https://example.com/") == 8
