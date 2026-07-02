# Política de Privacidade e Retenção de Dados — NPChat

Este documento descreve, para efeitos de RGPD e do Regulamento (UE) 2024/1689
(EU AI Act), que dados o NPChat processa, onde ficam armazenados, por quanto
tempo, e quais são os direitos dos titulares dos dados.

## 1. Responsável pelo tratamento

Near Partner. Para questões sobre esta política ou para exercer direitos
RGPD (acesso, retificação, apagamento), contactar o responsável interno
designado pela Near Partner para este sistema.

## 2. Que dados são processados

O NPChat **não tem sistema de contas nem autenticação** — o acesso é anónimo.
Os dados processados são:

| Dado | Onde fica guardado | Finalidade |
|---|---|---|
| Texto da pergunta do utilizador | `data/query_logs.db` (SQLite, local) | Melhorar a qualidade das respostas e detetar lacunas de conhecimento |
| Texto da resposta gerada | `data/feedback.db` (apenas se houver feedback 👍/👎) | Avaliar e treinar a lógica de ranking de retrieval |
| Pontuações de retrieval, tempos de resposta | `data/query_logs.db` | Monitorização de desempenho |
| Feedback (👍/👎) | `data/feedback.db` | Ajuste automático de pesos de retrieval (não afeta indivíduos) |
| Endereço IP do cliente | Apenas em memória, nunca persistido em disco | Rate limiting (30 pedidos/minuto) — descartado após a janela de 60s |
| Respostas em cache | `data/response_cache.db` | Evitar chamadas repetidas ao LLM |

**Não são processados dados biométricos, de saúde, ou qualquer categoria
especial de dados**, exceto se um utilizador os incluir voluntariamente no
texto livre de uma pergunta — o sistema não solicita nem depende desse tipo
de informação.

## 3. Onde os dados são processados (residência de dados)

O NPChat corre **inteiramente em infraestrutura local** (LLM via llama.cpp,
embeddings via sentence-transformers em processo, base de dados ChromaDB e
SQLite locais). **Nenhum dado de utilizador é enviado para serviços de
terceiros ou para fora da máquina/rede onde a aplicação corre.** Não há
chamadas a APIs de IA externas (OpenAI, Anthropic, etc.) nem a serviços cloud
de inferência.

## 4. Retenção e apagamento

- Registos de queries (`query_logs.db`) e feedback (`feedback.db`) são
  conservados durante **90 dias** (`config.log_retention_days`), após os
  quais são apagados automaticamente por uma tarefa diária do agendador
  (`src/scheduler.py: run_log_retention_cleanup`, 03:30).
- Entradas de cache expiram e são limpas diariamente às 03:00
  (`run_cache_cleanup`).
- Para apagar dados antes do prazo de retenção (ex.: pedido de titular de
  dados), usar `QueryLogger.delete_older_than(0)` e
  `FeedbackStore.delete_older_than(0)`, ou apagar diretamente os ficheiros em
  `data/`.

## 5. Base legal e finalidade

O tratamento assenta no **interesse legítimo** da Near Partner em operar e
melhorar um assistente informativo sobre os seus próprios serviços. Não há
decisões automatizadas com efeitos jurídicos ou similarmente significativos
sobre indivíduos (Art. 22 RGPD) — o único ajuste automático é a ponderação
agregada de chunks de conteúdo consoante o feedback recebido, não a avaliação
de pessoas.

## 6. Transferências internacionais

Não aplicável — não há transferência de dados para fora da infraestrutura
local onde o NPChat corre (ver secção 3).

## 7. Direitos dos titulares

Qualquer pessoa que tenha interagido com o chatbot pode solicitar acesso,
retificação ou apagamento dos seus dados, contactando o responsável indicado
na secção 1. Como o sistema não associa queries a identidades, o pedido deve
incluir o texto aproximado e período da interação para permitir localizar os
registos relevantes.
