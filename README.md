# Multiagents Debate Writer

Sistema em Python para geracao de artigos em Markdown a partir de uma solicitacao do usuario, usando `LangGraph` como orquestrador principal, `AutoGen` como engine interna do debate e `OpenAI Responses API` para intake, pesquisa, sintese e redacao final.

## Visao geral

O objetivo do projeto e produzir artigos mais robustos do que um pipeline linear comum. Em vez de gerar texto diretamente apos a pesquisa, o fluxo passa por um debate estruturado entre tres agentes com papeis complementares:

- `Analitico`: organiza o raciocinio e a estrutura.
- `Critico`: tensiona exageros, riscos e lacunas.
- `Estrategico`: amplia contexto e relevancia pratica.

O debate nao e decorativo: ele alimenta a sintese editorial e influencia o outline e o artigo final.
O sistema agora escreve e revisa o artigo por secao. Cada secao do outline vira uma unidade propria de geracao, revisao e, se necessario, recuperacao focada com pesquisa adicional e mini-debate, preservando as secoes ja aprovadas.
O orquestrador tambem enriquece dinamicamente os prompts do debate com contexto do assunto, eixos de analise, termos-chave e instrucoes especializadas por agente, para que o debate se adapte ao dominio do pedido em vez de usar um prompt generico fixo.

## Arquitetura

### Principios

- `LangGraph` controla estado, fluxo, repeticao e persistencia.
- `AutoGen` fica encapsulado em `app/debate/autogen_runner.py`.
- O restante do sistema conversa com o debate apenas pelo contrato `DebateEngine.run(...)`.
- Tudo roda em um unico processo Python.
- A pesquisa e modular para futura troca de backend.

### Fluxo

```text
Usuario
  -> orchestrator_intake
  -> research_node
  -> debate_node
  -> debate_node (opcional, via edge condicional)
  -> synthesis_node
  -> section_init_node
  -> section_write_node
  -> section_review_node
  -> section_research_node (opcional, apenas para secao fraca)
  -> section_debate_node (opcional, apenas para secao fraca)
  -> section_write_node (revisita apenas a secao fraca)
  -> article_assembly_node
  -> save_file_node
  -> outputs/artigo_final.md
```

### Estrutura de pastas

```text
multiagents_debate_writer/
|- app/
|  |- graph/
|  |  |- state.py
|  |  |- nodes.py
|  |  |- build_graph.py
|  |  `- edges.py
|  |- debate/
|  |  |- autogen_runner.py
|  |  |- agents.py
|  |  |- prompts.py
|  |  `- formatter.py
|  |- research/
|  |  |- web_search.py
|  |  |- parser.py
|  |  `- summarizer.py
|  |- writer/
|  |  |- outline.py
|  |  |- article_writer.py
|  |  `- markdown_saver.py
|  |- llm/
|  |  |- openai_client.py
|  |  `- models.py
|  `- main.py
|- outputs/
|  `- artigo_final.md
|- tests/
|  |- test_graph.py
|  |- test_debate.py
|  `- test_writer.py
|- .env.example
|- pyproject.toml
`- README.md
```

## Configuracao

Copie `.env.example` para `.env` e ajuste os valores:

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

## Como rodar

No diretorio `multiagents_debate_writer/`:

```bash
python -m app.main --request "Crie um artigo sobre inteligencia artificial"
```

Opcionalmente, sobrescreva o caminho de saida:

```bash
python -m app.main --request "Crie um artigo sobre inteligencia artificial" --output-path outputs/ia.md
```

Para aumentar a verbosidade dos logs no terminal:

```bash
python -m app.main --request "Crie um artigo sobre inteligencia artificial" --log-level DEBUG
```

Se quiser trocar o estilo de log por ambiente:

```bash
set LOG_STYLE=utf8_blocks
python -m app.main --request "Crie um artigo sobre inteligencia artificial"
```

## Como funciona

### 1. orchestrator_intake

Interpreta a solicitacao do usuario e define:

- request normalizada
- objetivo editorial
- publico-alvo
- tom
- dominio do conhecimento
- lente disciplinar
- exigencias de evidencia
- pacote de enriquecimento do debate com:
  - contexto compartilhado do tema
  - eixos prioritarios de analise
  - termos-chave
  - especializacao adicional para `Analitico`, `Critico` e `Estrategico`
- restricoes
- queries de pesquisa
- prompt base do debate

### 2. research_node

Usa `OpenAI Web Search` para pesquisar o tema, deduplicar fontes e produzir:

- `research_notes`
- `research_summary`
- `source_references`
- `evidence_is_sufficient`
- `evidence_confidence`
- `evidence_gaps`
- `follow_up_queries`

### 3. debate_node

Chama `DebateEngine`, que executa o debate com `AutoGen` em rodadas:

- rodada 1: tese inicial
- rodada 2: critica cruzada
- rodada 3: refinamento final

Se ainda houver questoes em aberto, o grafo pode repetir `debate_node` ate o teto configurado.
Durante o debate, o terminal exibe logs com a rodada, o agente, o posicionamento extraido da fala e o texto emitido por cada especialista.
Os prompts dos agentes nao sao estaticos: o `orchestrator_intake` detecta o assunto do pedido e injeta contexto especializado para que os debatedores mudem o foco conforme o tema, como meio ambiente, saude, direito, educacao, economia ou outros dominios.

Se a pesquisa nao sustentar um artigo academico-profissional longo com rigor suficiente, o fluxo aborta a escrita do artigo e gera um relatorio formal de insuficiencia de evidencia.

### 4. synthesis_node

Consolida pesquisa + debate em um outline final coerente.

### 5. section_init_node

Transforma o outline estruturado em uma lista ordenada de `SectionUnit`, com:

- identificador estavel
- heading
- finalidade
- bullets
- tipo (`short_form` ou `standard`)
- meta de palavras
- estado de processamento

### 6. section_write_node

Redige apenas a secao atual, usando:

- contexto global de pesquisa e debate
- finalidade da secao
- pesquisas e mini-debates focados daquela secao, quando existirem
- feedback das revisoes anteriores da propria secao

### 7. section_review_node

Revisa apenas a secao atual com rubrica adequada ao tipo:

- `short_form` para `Resumo` e `Palavras-chave`
- `standard` para secoes analiticas

Se a secao falhar e ainda houver budget, o fluxo entra em recuperacao focada apenas para ela. Se o budget esgotar, a secao e aceita com alerta explicito de qualidade.

### 8. section_research_node e section_debate_node

Quando uma secao falha, o sistema:

- gera um `SectionRecoveryPlan`
- executa queries focadas apenas naquela secao
- roda um mini-debate especializado apenas naquela secao
- volta para reescrever somente a secao fraca

Secoes aprovadas ficam congeladas e nao sao regeneradas por falhas em outras partes do artigo.

### 9. article_assembly_node

Concatena as secoes aprovadas na ordem do outline, inclui referencias uma unica vez no final e, se houver secoes aceitas com ressalvas, acrescenta `## Alertas de Qualidade`.

### 10. save_file_node

Cria a pasta de saida quando necessario e salva o arquivo final.

## Testes

Execute a suite com:

```bash
pytest
```

Os testes usam mocks/fakes para evitar chamadas reais a OpenAI e a pesquisa web.

## Limitacoes atuais

- O backend de pesquisa padrao depende do `web_search` da OpenAI.
- O debate adaptativo esta limitado a no maximo 5 rodadas.
- A recuperacao focada por secao esta limitada ao teto configurado em `SECTION_RETRY_MAX`.
- O CLI atual e simples e nao expoe checkpoint persistente.
- A versao inicial produz artigo academico em Markdown ou relatorio de insuficiencia de evidencia em Markdown.

## Roadmap

- adicionar checkpoint/persistencia do grafo
- suportar human-in-the-loop
- adicionar votacao ou arbitragem entre agentes
- suportar multiplos formatos de saida
- suportar outros tipos de documento alem de artigo
- incluir observabilidade e tracing mais detalhados
