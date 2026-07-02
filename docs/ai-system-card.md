# Ficha do Sistema de IA — NPChat

Documento técnico de suporte à conformidade com o Regulamento (UE) 2024/1689
(EU AI Act), Artigos 4.º (literacia em IA) e 50.º (transparência). Mantido
pela Near Partner enquanto **implementadora (deployer)** deste sistema.

## 1. Identificação e finalidade

- **Nome:** NPChat — Near Partner RAG Chatbot
- **Finalidade prevista:** responder a perguntas sobre os serviços, cultura,
  valores e equipa da Near Partner, com base em conteúdo público extraído de
  nearpartner.com (blog e páginas institucionais).
- **Utilização fora de âmbito (não suportada):** aconselhamento jurídico,
  financeiro, médico ou de RH; decisões sobre pessoas (contratação, crédito,
  avaliação de desempenho); qualquer uso que envolva dados biométricos ou
  categorias especiais de dados.
- **Público-alvo:** visitantes/clientes que interagem com o chatbot para se
  informar sobre a Near Partner. Acesso anónimo, sem contas de utilizador.

## 2. Classificação de risco (EU AI Act)

**Risco limitado** (não é um sistema de alto risco do Anexo III — não toma
nem apoia decisões sobre emprego, crédito, educação, aplicação da lei,
migração, biometria, ou acesso a serviços essenciais).

Está sujeito às obrigações de transparência do **Artigo 50.º**, por interagir
diretamente com pessoas: a interface informa o utilizador de que está a
falar com um sistema de IA (ver `app/main_app.py`, aviso no separador Chat).

A Near Partner atua como **implementadora**, não como fornecedora do modelo
de base — usa o modelo aberto Qwen2.5-7B-Instruct sem fine-tuning nem
modificação substancial. Não há, por isso, obrigações de fornecedor de
modelo de IA de finalidade geral (GPAI) a cumprir por parte da Near Partner.

## 3. Modelo e arquitetura

| Componente | Detalhe |
|---|---|
| LLM de geração | Qwen2.5-7B-Instruct-Q4_K_M, servido localmente via llama.cpp (API compatível com OpenAI) |
| Embeddings | `mxbai-embed-large-v1` (sentence-transformers, em processo) |
| Reranking | FlashRank (cross-encoder) |
| Retrieval | Pesquisa híbrida (semântica + BM25/RRF), reranking, opcionalmente multi-query/HyDE |
| Armazenamento vetorial | ChromaDB local |
| Infraestrutura | 100% local — sem chamadas a APIs de IA de terceiros, sem RunPod ou outro backend cloud em uso |

## 4. Dados de treino / conhecimento

O modelo de base (Qwen2.5-7B-Instruct) é pré-treinado por terceiros (Alibaba
Cloud/Qwen team); a Near Partner não o treina nem ajusta. A base de
conhecimento usada em runtime (RAG) é composta exclusivamente por conteúdo
público já publicado pela própria Near Partner em nearpartner.com.

## 5. Limitações conhecidas

- **Alucinação:** como qualquer LLM, pode gerar informação incorreta ou
  inventada, sobretudo quando a base de conhecimento não cobre a pergunta.
  Mitigado por: aviso de baixa confiança na UI, prompt que instrui o modelo a
  admitir desconhecimento, e citação de fontes.
- **Confiança automática não é garantia:** a auto-avaliação de confiança
  (0–1) é feita pelo próprio LLM e é indicativa, não uma métrica auditada
  externamente.
- **Idiomas:** respostas fiáveis apenas em português e inglês (espelha o
  idioma da pergunta); comportamento não testado noutros idiomas.
- **Desempenho:** hardware de inferência de single-slot; respostas podem
  demorar 15–30s.
- **Injeção de prompt:** mitigada (sanitização de input, hardening do system
  prompt) mas não eliminada — tratar respostas como não fiáveis para decisões
  críticas.

## 6. Supervisão humana

Ver [`docs/human-oversight.md`](human-oversight.md) para o processo de
revisão humana (queries de baixa confiança, feedback negativo repetido,
correção da base de conhecimento).

## 7. Literacia em IA (Art. 4.º)

Pessoal da Near Partner que administra o NPChat (gestão da base de
conhecimento, revisão de queries sinalizadas, configuração de definições no
separador *Settings*) deve estar familiarizado com:
- O funcionamento geral de sistemas RAG e as suas limitações (alucinação,
  dependência da qualidade do retrieval).
- Este documento e o `PRIVACY.md`.
- O processo de supervisão humana descrito em `docs/human-oversight.md`.

## 8. Contacto

Para questões técnicas ou de conformidade sobre este sistema, contactar o
responsável interno designado pela Near Partner.
