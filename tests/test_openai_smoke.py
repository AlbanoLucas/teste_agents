import os

import pytest

from app.llm.models import IntakePlan
from app.llm.openai_client import OpenAIResponsesClient
from app.prompts.builders import build_intake_prompt


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Smoke test opt-in: defina OPENAI_API_KEY para validar a integracao real.",
)


def test_openai_smoke_can_generate_an_intake_plan() -> None:
    client = OpenAIResponsesClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    payload = client.generate_structured(
        schema_model=IntakePlan,
        envelope=build_intake_prompt("Escreva um artigo sobre transicao energetica"),
        temperature=0.2,
    )

    assert payload.normalized_request
    assert payload.research_queries
