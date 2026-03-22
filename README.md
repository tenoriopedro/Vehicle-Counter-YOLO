# Sistema de Contagem e Rastreamento de Veículos (YOLOv8)

<p align="center">
  <img src="https://github.com/tenoriopedro/Vehicle-Counter-YOLO/tree/main/assets/video_counter_car.gif?raw=true" alt="Demonstração do Contador de Veículos YOLOv8" width="700"/>
</p>

---

## 🚀 Visão Geral

Este projeto utiliza o **YOLOv8** para detetar, rastrear e **contar veículos** (carros, motas, camiões) em vídeos de tráfego.

A lógica principal não se limita a detetar, mas aplica um rastreador (tracker) para identificar objetos únicos e **contabilizar os veículos de forma separada por direção**, com base em zonas de "entrada" e "saída" predefinidas no vídeo.

### 🎯 Funcionalidades

* **Deteção e Rastreamento:** Identifica e segue veículos usando YOLOv8.
* **Contagem por Classe:** Contabiliza 3 classes: Carros, Motocicletas e Camiões.
* **Contagem Direcional:** Regista a contagem separadamente para veículos que "entram" vs. "saem" da zona monitorizada.
* **Resultado em Vídeo:** Gera um novo ficheiro de vídeo com os contadores, zonas e caixas de deteção sobrepostos.

---

# DONT COPY THIS CODE. ITS UNDER RECONSTRUCTION

<!-- ### 🛠️ Stack Tecnológico

* **Python**
* **YOLOv8 (Ultralytics)**
* **OpenCV** (Processamento de vídeo, desenho de zonas)
* **NumPy** (Cálculos de zona/contagem)

---

### ⚙️ Como Executar (Localmente)

<details>
  <summary>
    <strong>[+] Clique para expandir</strong> (Instruções de instalação e execução)
  </summary>
  
  <p>O projeto inclui um vídeo de teste (<code>track_video_car01.mp4</code> na pasta <code>test_files/</code>) para que possa ser executado imediatamente.</p>

  <h4>1. Clone o repositório</h4>
  <pre><code>git clone https://github.com/tenoriopedro/Vehicle-Counter-YOLO.git
cd Vehicle-Counter-YOLO</code></pre>

  <h4>2. Crie e ative um ambiente virtual</h4>
  <pre><code>python -m venv venv
.\venv\Scripts\activate.ps1  # Windows
source venv/bin/activate    # Linux/Mac</code></pre>

  <h4>3. Instale as dependências</h4>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h4>4. Execute os scripts</h4>
  <p>O processo é feito em duas etapas:</p>
  <ol>
    <li><strong>Processar o vídeo</strong> (Aplica o YOLO e o tracker):</li>
    <pre><code>python compile_video.py</code></pre>
    <li><strong>Gerar o resultado final</strong> (Renderiza o vídeo processado):</li>
    <pre><code>python show_results.py</code></pre>
  </ol>
  
  <p>O vídeo final (<code>video_countingCar_result01.mp4</code>) será guardado na pasta <code>result_files/</code>.</p>
</details> -->
