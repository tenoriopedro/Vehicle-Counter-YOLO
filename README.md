# 🚦 Pipeline de Telemetria de Tráfego (YOLOv8 + OpenVINO + PyArrow)

## 🚀 Visão Geral

Este sistema atua como um produtor de telemetria de tráfego otimizado para ambientes de recursos limitados (_Edge Computing_). Em vez de focar na renderização visual, o sistema extrai dados de movimento em tempo real e escoa-os para um _Data Lake_ local sem saturação de memória.

A arquitetura baseia-se numa separação estrita de responsabilidades:

| Camada                        | Tecnologia         | Responsabilidade                                                                         |
| :---------------------------- | :----------------- | :--------------------------------------------------------------------------------------- |
| **Visão (Produtor)**          | YOLOv8 + OpenVINO  | Rastreamento vetorial, desviando a carga matemática do CPU para a GPU integrada.         |
| **Domínio (Contratos)**       | Pydantic           | Garantia de tipagem estrita de todos os eventos de tráfego, prevenindo estruturas órfãs. |
| **I/O (Escoamento)**          | PyArrow            | _Buffer_ de memória com escrita direta em disco no formato colunar **Parquet**.          |
| **Analytics (Processamento)** | Agregação em Lotes | Motor _out-of-core_ que analisa ficheiros em disco sem risco de falha de memória (OOM).  |

### 🎯 Funcionalidades

- Extração espacial e classificação estruturada (Carros, Motociclos, Autocarros, Camiões).
- Validação de fluxo Norte/Sul através da transição do eixo Y transversal.
- Gestão de memória térmica: ingestão direta em disco (_flush_ de lotes otimizado para compressão colunar PyArrow).
- Interface de calibração espacial estática (OpenCV) para definição da barreira lógica.

---

## ⚙️ Arquitetura Alvo e Instalação

Este projeto foi desenhado com foco em resiliência para **Edge Computing**, otimizado especificamente para hardware com recursos limitados (CPUs Intel sem gráfica dedicada). O código tira partido da aceleração OpenVINO para maximizar o rendimento termodinâmico em equipamentos modestos.

O pipeline Python é encapsulado pelo [uv](https://github.com/astral-sh/uv), exigindo Python 3.11.9+

### 1. Perfil Principal: Linux / Hardware Intel (Recomendado)

Para ativar a aceleração nativa na GPU integrada (Iris Xe / UHD) e evitar o estrangulamento do processador na descodificação H.264, o sistema operativo requer os seguintes _drivers_:

```bash
sudo apt update
sudo apt install intel-opencl-icd ffmpeg
```

### 2. Perfil Universal: Mac, Windows ou NVIDIA

Se não opera num ecossistema Intel/Linux, ignore a instalação de dependências de sistema operativo acima. O instalador `uv` encarrega-se do ambiente Python.

**Nota Arquitetónica:** Nestes sistemas, o modelo `yolov8n_openvino_model` fornecido no comando da Fase 2 poderá reverter automaticamente para inferência via CPU puro. Para utilizar aceleração dedicada (CUDA ou MPS), deve fornecer um modelo nativo PyTorch (`.pt`) no argumento `--m`.

### 3. Sincronização do Ambiente (Comum a todos)

Executa a clonagem do repositório e a sincronização rigorosa do ambiente virtual num único fluxo:

```bash
git clone [https://github.com/tenoriopedro/vehicle-counter-YOLO.git](https://github.com/tenoriopedro/vehicle-counter-YOLO.git)
cd vehicle-counter
uv sync
```

## 🛠️ Fluxo de Execução (CLI)

O pipeline divide-se em três fases distintas. Todos os comandos devem ser executados através do `uv run` para garantir o encapsulamento do ambiente.

### Fase 1: Calibração da Linha de Contagem

Inicia a janela interativa redimensionável. Defina a linha delimitadora clicando em dois pontos da via e prima `q` para guardar a matriz.

```bash
uv run vehicle-calibrate -v <caminho/para/o/video.mp4>
```

_(Gera um ficheiro local `.json` com as coordenadas espaciais)_.

### Fase 2: Inferência Otimizada (GPU)

Inicia o rastreamento direcionado para a infraestrutura OpenVINO. Os frames visuais são descartados, e a telemetria é cimentada no diretório `.tmp_`, sendo fundida em Parquet no término.

```bash
uv run vehicle-counter -m <caminho/para/yolov8n_openvino_model> -v <caminho/para/o/video.mp4> -o <caminho/para/diretorio_saida>
```

### Fase 3: Extração Analítica

Invoca o motor de agregação para ler e contabilizar os tensores do ficheiro `.parquet`, reportando o caudal por classe e direção lógica.

```bash
uv run vehicle-report
```
