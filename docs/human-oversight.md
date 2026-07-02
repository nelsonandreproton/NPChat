# Supervisão Humana — NPChat

Este documento formaliza, para efeitos do Artigo 26.º do EU AI Act
(obrigações de implementadores), o processo de supervisão humana já suportado
pelas funcionalidades existentes de feedback e analytics do NPChat.

## 1. Mecanismos automáticos que despoletam revisão humana

| Gatilho | Onde aparece | Ação automática |
|---|---|---|
| Confiança da resposta < 0.5 (quando "Auto-Quality Evaluation" está ativo) | Aviso `⚠️` no chat | Nenhuma ação automática além do aviso ao utilizador |
| Pontuação média de retrieval baixa | Separador **Analytics > Low Score** | Sinaliza lacunas de conhecimento para revisão |
| 2 ou mais 👎 na mesma query | Separador **Analytics > Learning > Flagged Queries** | Query marcada como `pending`, chunks penalizados, cache invalidada |
| Feedback 👎 isolado | Separador **Analytics > Negative** | Registado para análise de tendências |

## 2. Processo de revisão

1. **Cadência:** a pessoa responsável pela gestão de conteúdo do NPChat
   (designada pela Near Partner) deve rever o separador **Analytics >
   Learning > Flagged Queries** e **Analytics > Low Score** com regularidade
   (recomendado: semanalmente, alinhado com o relatório automático de
   domingo às 23:00 gerado por `src/scheduler.py: generate_weekly_report`).
2. **Para cada query sinalizada**, o revisor:
   - Confirma se a resposta estava de facto incorreta ou incompleta.
   - Se a base de conhecimento tiver uma lacuna, atualiza o conteúdo fonte em
     nearpartner.com (ou o processo de scraping/ingestão) e volta a ingerir.
   - Marca a query como **Resolvida** ou **Dispensada** no próprio separador
     (`FeedbackLearner.resolve_flag`).
3. **Casos de erro sistemático** (várias queries sinalizadas sobre o mesmo
   tema): tratar como prioridade — pode indicar contexto em falta na base de
   conhecimento ou um problema no prompt/sistema, não apenas conteúdo.

## 3. Limites da supervisão automática

O ajuste automático de pesos de chunks (👍 +0.1 / 👎 −0.15) e a invalidação de
cache são **medidas de mitigação imediata**, não substituem a revisão
humana — não corrigem a causa raiz (conteúdo em falta ou impreciso) nem
avaliam se o feedback do utilizador estava correto.

## 4. Intervenção manual disponível

A pessoa responsável pode a qualquer momento, via separador **Settings**:
- Desativar funcionalidades experimentais (HyDE, query expansion, etc.).
- Forçar novo scraping + ingestão de conteúdo.
- Limpar a cache de respostas.
- Ajustar `top_k`/temperatura para tornar as respostas mais conservadoras.

## 5. Responsabilidade

A Near Partner designa internamente quem desempenha este papel de revisão;
este documento não impõe um nome específico, apenas o processo a seguir. Ver
também [`docs/ai-system-card.md`](ai-system-card.md) secção 7 (literacia em
IA) para os conhecimentos mínimos esperados de quem desempenha esta função.
