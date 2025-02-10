**Importante sobre a tradução**

O texto abaixo foi traduzido usando ferramentas de IA (tradução automática). Como esse processo pode conter erros ou imprecisões, recomendamos consultar a versão original em inglês ou espanhol para garantir a precisão das informações.

---

[![English](https://img.shields.io/badge/lang-English-blue)](README.en.md)
[![Español](https://img.shields.io/badge/lang-Español-purple)](README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](README.zh.md)
[![Português](https://img.shields.io/badge/lang-Português-brightgreen)](README.pt.md)
[![Deutsch](https://img.shields.io/badge/lang-Deutsch-blueviolet)](README.de.md)
[![Italiano](https://img.shields.io/badge/lang-Italiano-orange)](README.it.md)
[![日本語](https://img.shields.io/badge/lang-日本語-yellowgreen)](README.jp.md)
[![العربية](https://img.shields.io/badge/lang-العربية-lightgrey)](README.ar.md)
[![עברית](https://img.shields.io/badge/lang-עברית-teal)](README.he.md)
[![Русский](https://img.shields.io/badge/lang-Русский-lightblue)](README.ru.md)
[![Українська](https://img.shields.io/badge/lang-Українська-skyblue)](README.uk.md)

# YoloMOT
Este repositório contém código para converter as previsões do Ultralytics YOLO para o formato MOT-Challenge. Também inclui um script para gerar um conjunto de dados sintético de ground truth para rastreamento, juntamente com os resultados de rastreamento correspondentes, possibilitando testes rápidos com o TrackEval.

## Explicação das Funções Utilitárias
A funcionalidade principal é implementada no arquivo `utils.py`.
Você pode salvar as previsões produzidas por um modelo YOLO diretamente no formato MOT-Challenge utilizando a função `save_mot_results`. Alternativamente, se preferir trabalhar com arquivos JSON, você pode primeiro obter um arquivo JSON de previsões usando a função `save_results_as_json` e, em seguida, converter esse JSON para o formato MOT-Challenge com `save_mot_from_json`.

### Salvar Previsões Diretamente no Formato MOT
Para salvar as previsões, o código as recupera utilizando o método `Results.to_df()` para cada frame e as escreve no formato MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
Neste formato, o campo `id` é definido como `-1` para detecções que ainda não foram rastreadas. Para dados de ground truth, você pode, por exemplo, definir o valor de `conf` para `1`.
```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```
A seguir, um exemplo de saída do uso do método `to_df()` em um objeto de Resultados do Ultralytics:
```bash
       name  class  confidence  \
0    person      0     0.92043   
1    person      0     0.91078   
2    person      0     0.85290   
3  backpack     24     0.60395   
                                         box  track_id  \
0  {'x1': 1642.54541, 'y1': 597.04321, 'x2': 2352...         1   
1  {'x1': 92.58405, 'y1': 7.53223, 'x2': 727.1163...         2   
2  {'x1': 2227.49854, 'y1': 1024.51343, 'x2': 291...         3   
3  {'x1': 92.28073, 'y1': 203.9021, 'x2': 445.741...         4   
                                         segments   
0  {'x': [2262.0, 2256.0, 2256.0, 2262.0, 2274.0,...   
1  {'x': [228.0, 228.0, 216.0, 210.0, 204.0, 186....   
2  {'x': [2652.0, 2652.0, 2646.0, 2646.0, 2640.0,...   
3  {'x': [204.0, 210.0, 222.0, 222.0, 234.0, 240....   
```

Tanto `save_mot_results` quanto `save_mot_from_json` podem salvar também rótulos de segmentação no formato MOTS, se solicitado. O formato MOTS é definido da seguinte forma:
```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

Neste formato, `rle` significa codificação Run-Length, tal como é usada no conjunto de dados COCO. Consequentemente, a biblioteca pycocotools é necessária para codificar e decodificar as máscaras neste formato.

### Salvar Previsões como JSON para Uso Futuro
O arquivo JSON contém as previsões em formato JSON, geradas utilizando o método `Results.to_json()` para cada frame.
```json
{
    "0": [
        {
            "name": "person",
            "class": 0,
            "confidence": 0.91342,
            "box": {
                "x1": 2751.35596,
                "y1": 243.29077,
                "x2": 3436.21362,
                "y2": 2054.81641
            },
            "track_id": 1,
            "segments": {
                "x": [2868.0, 2868.0, 2862.0, ..., 3048.0],
                "y": [252.0, 270.0, 276.0, ..., 252.0]
            }
        },
        {
            "name": "skateboard",
            "class": 36,
            "confidence": 0.39874,
            "box": {
                "x1": 3746.27856,
                "y1": 1691.79749,
                "x2": 3838.45679,
                "y2": 1793.54187
            },
            "track_id": 5,
            "segments": {
                "x": [3756.0, 3756.0, 3774.0, ..., 3834.0],
                "y": [1692.0, 1734.0, 1734.0, ..., 1692.0]
            }
        }
    ],
    "1": [
        {
            "name": "person",
            "class": 0,
            "confidence": 0.93321,
            "box": {
                "x1": 3064.85596,
                "y1": 260.86932,
                "x2": 3836.20898,
                "y2": 2044.81006
            },
            ...
        }
    ],
    ...
}
```

## Como Usar o TrackEval
Para avaliar os resultados de rastreamento, comece clonando o repositório original do [TrackEval](https://github.com/JonathonLuiten/TrackEval/):
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

Em seguida, copie o diretório `data` corretamente configurado —que deve conter seus dados de ground truth e as previsões para cada tracker— para dentro do diretório TrackEval. Para criar uma coleção sintética de previsões de trackers e ground truths, você pode usar o script de exemplo `synthetic_dataset.py`.

Uma vez que tudo esteja configurado, execute o script `run_mot_challenge.py` com os argumentos de linha de comando apropriados para avaliar seu conjunto de dados customizado. Este script computa métricas de rastreamento como HOTA, MOTA, etc. Para mais detalhes, consulte a documentação do TrackEval.

**Nota:** Certifique-se de definir `--DO_PREPROC` para `False` para evitar problemas potenciais. Por exemplo:
```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## Formato de Avaliação MOT Challenge
**Caminho do Ground Truth:**  
`/data/gt/mot_challenge/<SeuChallenge>-<eval>/<SeqName>/gt/gt.txt`    
Este arquivo contém os rótulos de ground truth no formato MOT-Challenge. Aqui, `<SeuChallenge>` é o nome do seu desafio ou conjunto de dados, `<eval>` pode ser `train`, `test` ou `all`, e `<SeqName>` corresponde à sequência de vídeo.

A estrutura de diretórios sob `/data/gt/mot_challenge/<SeuChallenge>-<eval>` deve ser a seguinte:
```
.
|—— <SeqName01>
    |—— gt
        |—— gt.txt
    |—— seqinfo.ini
|—— <SeqName02>
    |—— ...
|—— <SeqName03>
    |—— ...
```

**Exemplo de arquivo `seqinfo.ini`:**
```
[Sequence]
name=<SeqName>
imDir=img1
frameRate=30
seqLength=525
imWidth=1920
imHeight=1080
imExt=.jpg
```

**Arquivos de Sequência:**  
Dentro da pasta `/data/gt/mot_challenge/seqmaps`, crie arquivos de texto listando os nomes das sequências. Por exemplo, crie `<SeuChallenge>-all.txt`, `<SeuChallenge>-train.txt` e `<SeuChallenge>-test.txt` com a seguinte estrutura:
```
<SeuChallenge>-all.txt
name
<seqName1>
<seqName2>
<seqName3>
<SeuChallenge>-train.txt
name
<seqName1>
<seqName2>
<SeuChallenge>-test.txt
name
<seqName3>
```

**Caminho das Previsões:**  
`/data/trackers/mot_challenge/<SeuChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`   
Este arquivo armazena as previsões para cada sequência de vídeo para cada tracker. Aqui, `<TrackerName>` é o nome do tracker ou algoritmo utilizado.

A estrutura de diretórios sob `/data/trackers/mot_challenge/<SeuChallenge>-<eval>` deve ser a seguinte:
```
.
|—— <Tracker01>
    |—— data
        |—— <SeqName01>.txt
        |—— <SeqName02>.txt
|—— <Tracker02>
    |—— ...
|—— <Tracker03>
    |—— ...
```
