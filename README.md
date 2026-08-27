# 🚦 Pipeline de Telemetria de Tráfego (YOLOv8 + OpenVINO + PyArrow)

## 🚀 Visão Geral

Este sistema atua como um produtor e consumidor de telemetria de tráfego, desenhado sob a topologia de um _Data Lake_ moderno. Em vez de escrever ficheiros estáticos localmente, o sistema extrai dados de movimento vetorial em tempo real, emite-os para um _Message Broker_ distribuído e ingere-os na nuvem em lotes otimizados

A arquitetura baseia-se numa separação estrita de responsabilidades:

| Camada                    | Tecnologia        | Responsabilidade                                                                         |
| :------------------------ | :---------------- | :--------------------------------------------------------------------------------------- |
| **Visão (Produtor)**      | YOLOv8 + OpenVINO | Rastreamento espacial e inferência, com escoamento assíncrono para a placa de rede.      |
| **Rede (Broker)**         | Confluent Kafka   | Absorção de picos de tráfego e imobilização cronológica de eventos (Tópico).             |
| **Ingestão (Consumidor)** | PyArrow + Boto3   | Subscrição de rede, acumulação na memória RAM e conversão colunar para a nuvem (AWS S3). |
| **Domínio (Contratos)**   | Pydantic          | Garantia de tipagem estrita de todos os eventos JSON, prevenindo estruturas órfãs.       |

### 🎯 Funcionalidades

- Extração espacial e classificação estruturada (Carros, Motociclos, Autocarros, Camiões).
- Streaming assíncrono: O motor visual nunca bloqueia à espera de I/O, garantindo FPS máximo.
- Gestão de rede resiliente: Retenção de eventos não entregues no _buffer_ C do Kafka em caso de falhas.
- Arquitetura _Cloud-Native_: A agregação ocorre em memória com injeção direta no armazenamento de objetos (S3) no formato comprimido **Parquet**

---

## ⚙️ Arquitetura Alvo e Instalação

O pipeline é encapsulado pelo [uv](https://github.com/astral-sh/uv), exigindo Python 3.11.9+

### 1. Perfil Principal: Linux / Hardware Intel (Recomendado)

Para ativar a aceleração nativa na GPU integrada e evitar o estrangulamento do processador na descodificação H.264:

```bash
sudo apt update
sudo apt install intel-opencl-icd ffmpeg
```

_(Se utiliza Mac, Windows ou gráfica NVIDIA, ignore este passo; o instalador `uv` encarrega-se do ambiente)._

### 2. Sincronização do Ambiente

Executa a clonagem do repositório e a sincronização rigorosa do ambiente virtual:

```bash
git clone https://github.com/tenoriopedro/vehicle-counter-YOLO.git
cd vehicle-counter
uv sync
```

### 3. Cofre de Credenciais e Infraestrutura

A escrita direta na nuvem exige autorização programática. Crie um ficheiro `.env` na raiz do projeto (nunca comitado) com a seguinte estrutura:

```ini
AWS_ACCESS_KEY_ID=sua_access_key
AWS_SECRET_ACCESS_KEY=sua_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=seu-bucket-de-telemetria
```

## 📂 Contrato de Dados (Obrigatório)

O vídeo fonte deve ser isolado num **Contexto** dentro do diretório `data/raw/`. A estrutura não tolerará ficheiros avulsos.

```text
data/raw/
└── avenida_norte/
    └── avenida_norte.mp4  <-- O vídeo tem de assumir este nome exato
```

## 🛠️ Fluxo de Execução (Microsserviços)

A arquitetura exige a execução de processos isolados. Levante primeiro a infraestrutura de rede:

```bash
docker-compose up -d
```

### Serviço A: Orquestrador Visual (O Produtor)

Inicia a janela interativa de calibração espacial, seguida do rastreamento direcionado. A telemetria é emitida para a porta `9092` no tópico `vehicle-telemetry`.

```bash
# 1. Calibrar a via
uv run vehicle-calibrate --context <nome_do_contexto>

# 2. Iniciar a emissão (Streaming)
uv run vehicle-counter --context <nome_do_contexto> -m <caminho/para/modelo>
```

### Serviço B: Motor de Ingestão (O Consumidor)

Num terminal paralelo, inicie o _daemon_ que escuta a placa de rede, agrupa os eventos em _micro-batches_ e converte-os em tabelas Parquet que são enviadas para o S3.

```bash
# (Comando provisório de desenvolvimento)
uv run python src/vehicle_counter/consumer/kafka_to_s3.py
```
