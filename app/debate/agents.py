"""Definitions for the specialist debate agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DebateAgentSpec:
    """Prompt-level identity for a debate participant."""

    name: str
    responsibility: str
    objective: str
    style: str
    max_words: int = 140


DEBATE_AGENTS: tuple[DebateAgentSpec, ...] = (
    DebateAgentSpec(
        name="Analitico",
        responsibility="Organizar o raciocinio e a estrutura do artigo.",
        objective=(
            "Propor uma linha logica clara, sugerir secoes, hierarquia e transicoes "
            "coerentes entre argumentos."
        ),
        style="Direto, estruturado e orientado a clareza editorial.",
    ),
    DebateAgentSpec(
        name="Critico",
        responsibility="Tensionar exageros, lacunas e conclusoes superficiais.",
        objective=(
            "Questionar generalizacoes, apontar riscos e exigir rigor factual e "
            "nuance interpretativa."
        ),
        style="Cauteloso, incisivo e rigoroso.",
    ),
    DebateAgentSpec(
        name="Estrategico",
        responsibility="Expandir contexto, implicacoes e relevancia pratica.",
        objective=(
            "Conectar o tema a impacto social, economico e futuro, propondo framing "
            "relevante para o publico-alvo."
        ),
        style="Amplo, pragmatico e orientado a implicacoes.",
    ),
)
