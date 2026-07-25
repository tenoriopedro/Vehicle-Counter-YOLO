# 🚦 Pipeline de Telemetria de Tráfego (YOLOv8 + PyArrow)

> ⚠️ **AVISO DE RECONSTRUÇÃO:** Este projeto está a sofrer uma alteração arquitetónica profunda. Estamos a migrar de um script de processamento de vídeo monolítico para um pipeline de Engenharia de Dados resiliente.

---

## 🚀 Visão Geral (Novo Design)

Este sistema atua como um produtor de telemetria de tráfego pesado. Em vez de focar na renderização visual, o sistema extrai dados de movimento em tempo real e escoa-os para um *Data Lake* local.

A arquitetura atual baseia-se na separação estrita de responsabilidades:
1. **Visão (Produtor):** O YOLOv8 rastreia e identifica veículos, emitindo eventos puros.
2. **Domínio (Contratos):** Pydantic garante a tipagem rigorosa de todos os eventos de tráfego.
3. **I/O (Escoamento):** Um buffer de memória escreve os dados diretamente no disco em formato **Parquet** utilizando PyArrow, otimizado para consultas analíticas em DuckDB.

### 🎯 Funcionalidades Atuais
* Extração de matrizes de movimento e classificação (Carros, Motociclos, Camiões).
* Validação de direção (Norte/Sul) via interseção vetorial.
* Ingestão direta em disco sem dependência de Pandas (Baixo uso de RAM).