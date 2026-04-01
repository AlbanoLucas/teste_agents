# Multiagents Debate Writer

Sistema em Python para geracao de artigos em Markdown com `LangGraph` como orquestrador principal, `OpenAI Responses API` para intake, pesquisa, sintese e escrita, e `AutoGen` encapsulado apenas dentro do adapter de debate.

## Visao geral

O projeto foi desenhado para produzir artigos tecnico-cientificos mais robustos do que um pipeline linear de prompt unico. O fluxo combina:

- pesquisa web estruturada
- debate multiagente com papeis distintos
- sintese editorial
- escrita por secao
- revisao por secao
- recuperacao focada apenas das secoes fracas

O debate nao e decorativo: ele influencia o outline e a redacao final. As secoes aprovadas ficam congeladas e nao sao reescritas por falhas em outras partes do artigo.

## Principios arquiteturais

- `LangGraph` controla o fluxo, o estado serializavel, o roteamento e a ordem de execucao.
- `AutoGen` fica restrito a `app/debate/autogen_runner.py`.
- O restante do sistema conversa com o debate apenas pelo contrato `DebateEngine.run(...)`.
- O estado interno real do workflow vive em modelos Pydantic do pacote `app/workflow/`.
- `ProjectState` guarda apenas um campo `workflow`, que serializa o envelope tipado inteiro.
- Prompt engineering fica separado da execucao entre `app/prompts/builders.py` e `app/prompts/fragments.py`.
- Falhas, retries e classificacao de erro passam por `WorkflowErrorPolicy`.
- Tudo roda em um unico processo Python.

## Fluxo

```text
Usuario
  -> orchestrator_intake
  -> research_node
  -> debate_node
  -> debate_node (opcional, se ainda houver pendencias)
  -> synthesis_node
  -> section_init_node
  -> section_write_node
  -> section_review_node
  -> section_research_node (opcional, apenas para secao fraca)
  -> section_debate_node (opcional, apenas para secao fraca)
  -> section_write_node
  -> article_assembly_node
  -> failure_node (quando houver falha terminal)
  -> save_file_node
  -> outputs/artigo_final.md
```

Se a evidencia for insuficiente logo apos a pesquisa, o grafo desvia para `article_assembly_node` em modo `insufficiency_report`, sem entrar no debate ou na pipeline de secoes. Se qualquer etapa falhar, o grafo materializa um relatorio de falha em Markdown e ainda passa por `save_file_node`.

## Estrutura de pastas

```text
multiagents_debate_writer/
|- app/
|  |- debate/
|  |  |- agents.py
|  |  |- autogen_runner.py
|  |  |- formatter.py
|  |  `- prompts.py
|  |- graph/
|  |  |- build_graph.py
|  |  |- edges.py
|  |  |- nodes.py
|  |  |- services.py
|  |  |- state.py
|  |  `- handlers/
|  |- llm/
|  |  |- models.py
|  |  `- openai_client.py
|  |- prompts/
|  |  |- builders.py
|  |  |- fragments.py
|  |  `- __init__.py
|  |- research/
|  |  |- parser.py
|  |  |- summarizer.py
|  |  `- web_search.py
|  |- workflow/
|  |  |- error_policy.py
|  |  |- models.py
|  |  |- section_recovery.py
|  |  |- section_service.py
|  |  `- state_adapter.py
|  |- writer/
|  |  |- article_assembler.py
|  |  |- formatter.py
|  |  |- markdown_saver.py
|  |  |- outline.py
|  |  |- section_reviewer.py
|  |  `- section_writer.py
|  `- main.py
|- outputs/
|- tests/
|  |- test_graph_integration.py
|  |- test_openai_smoke.py
|  |- test_prompt_contracts.py
|  |- test_runtime_resilience.py
|  |- test_section_recovery.py
|  `- test_writer_components.py
|- .env.example
|- pyproject.toml
`- README.md
```

## Componentes principais

### `orchestrator_intake`

Interpreta a solicitacao do usuario e produz:

- request normalizada
- objetivo editorial
- publico
- tom
- dominio do conhecimento
- lente disciplinar
- exigencias de evidencia
- restricoes
- queries de pesquisa
- prompt base do debate
- pacote resolvido de enriquecimento dos agentes

### `research_node`

Executa buscas, normaliza fontes e consolida:

- `research_summary`
- `research_notes`
- `source_references`
- `evidence_is_sufficient`
- `evidence_confidence`
- `evidence_gaps`
- `follow_up_queries`

### `debate_node`

Executa o debate global com tres agentes:

- `Analitico`
- `Critico`
- `Estrategico`

O debate pode repetir via edge condicional ate o limite configurado quando ainda houver pendencias.

### `synthesis_node`

Transforma pesquisa + debate em um `OutlinePayload` estruturado.

### Pipeline por secao

- `section_init_node`: converte o outline em `SectionState`
- `section_write_node`: escreve apenas a secao atual
- `section_review_node`: revisa apenas a secao atual
- `section_research_node`: cria plano de recuperacao, executa pesquisa focada e resolve o `prompt_pack`
- `section_debate_node`: executa de fato o mini-debate focado da secao
- `article_assembly_node`: monta o Markdown final, referencias e alertas

### `failure_node`

Quando um handler falha, o workflow marca `status=failed`, registra `terminal_error` no envelope e roteia para `failure_node`, que gera um relatorio formal de falha em Markdown antes do salvamento final.

## Logs

Os logs foram estruturados para acompanhamento no terminal com blocos UTF-8 legiveis. Eles mostram:

- etapa atual do grafo
- queries de pesquisa
- decisao de suficiencia da evidencia
- rodadas do debate
- agente que falou
- posicionamento extraido da fala
- recuperacao focada por secao
- montagem e salvamento final

Os logs exibem a fala emitida pelos agentes, nao cadeia de pensamento privada.

## Configuracao

Copie `.env.example` para `.env` e ajuste:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OUTPUT_PATH=outputs/artigo_final.md
RESEARCH_MAX_SOURCES=6
DEBATE_MIN_ROUNDS=3
DEBATE_MAX_ROUNDS=5
SECTION_RETRY_MAX=3
LLM_TEMPERATURE=0.3
LOG_LEVEL=INFO
LOG_STYLE=utf8_blocks
ARTICLE_PROFILE=academic_rigid
ARTICLE_MIN_WORDS=6000
EVIDENCE_POLICY=abort
```

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Execucao

```bash
python -m app.main --request "Crie um artigo sobre inteligencia artificial"
```

Com sobrescrita do caminho de saida:

```bash
python -m app.main --request "Crie um artigo sobre inteligencia artificial" --output-path outputs/ia.md
```

Com mais verbosidade:

```bash
python -m app.main --request "Crie um artigo sobre inteligencia artificial" --log-level DEBUG
```

## Testes

Execute:

```bash
python -m pytest
```

Cobertura atual:

- contratos de prompt pack e schema estruturado
- estado aninhado em `workflow` e ausencia de aliases legados
- erro terminal persistido no estado e materializado em `failure_node`
- montagem final com referencias unicas e alertas de qualidade
- merge correto do contexto global na recuperacao de secao
- separacao entre `prepare_recovery(...)` e `run_section_debate(...)`
- integracao do grafo com retry por secao e roteamento de falha
- bypass do grafo para relatorio de insuficiencia
- fragments de prompt com omissao de blocos vazios e limites de truncamento
- smoke test opt-in com OpenAI real, executado apenas quando `OPENAI_API_KEY` estiver definido

## Limitacoes atuais

- O backend de pesquisa padrao depende do `web_search` da OpenAI.
- O debate global continua limitado ao teto configurado de rodadas.
- A recuperacao focada por secao continua limitada por `SECTION_RETRY_MAX`.
- O CLI atual ainda nao oferece checkpoint persistente nem human-in-the-loop.
- O smoke test real depende de credenciais e conectividade disponiveis no ambiente.

## Roadmap

- checkpoint persistente do grafo
- human-in-the-loop
- votacao ou arbitragem entre agentes
- formatos adicionais de saida
- tipos adicionais de documento alem de artigo
- tracing e observabilidade operacional mais profundos
