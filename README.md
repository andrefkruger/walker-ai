# Walker AI

Simulação de um boneco bípede aprendendo a andar usando **Reinforcement Learning (SAC)**, com física em Box2D, visualização em Pygame e **curriculum learning** progressivo por dificuldade de terreno.

## Destaques

- **SAC (Soft Actor-Critic)** como algoritmo de treino principal (há também suporte a PPO nas configurações)
- **Curriculum learning**: o terreno começa totalmente plano e vai ficando mais irregular conforme o agente atinge metas de distância
- Sistema de recompensas ajustado para evitar comportamentos degenerados (ex: penalidade por inclinação do torso, para impedir que o agente aprenda a "cair de lado" como estratégia)
- Visualização em tempo real do treinamento com Pygame

## Estrutura do projeto

```
.
├── agent/            # Implementação do agente SAC (rede neural, replay buffer)
├── environment/       # Ambiente de simulação física (Box2D)
├── render/            # Visualização com Pygame
├── config.py          # Todos os hiperparâmetros (física, rede, recompensas, curriculum)
├── train.py           # Script principal de treino
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
python train.py               # treina do zero com visualização
python train.py --no-render   # treina sem janela (mais rápido)
python train.py --load        # continua treinando a partir do melhor checkpoint salvo
python train.py --watch       # apenas assiste ao agente já treinado, sem treinar
```

## Como funciona o curriculum

O treino avança de fase automaticamente quando o agente atinge uma distância mínima de forma consistente:

| Fase | Terreno | Avança com |
|------|---------|------------|
| 0 | Plano | 2.0 m |
| 1 | Micro-ondulações | 2.0 m |
| 2 | Suave | 2.5 m |
| 3 | Médio | 3.0 m |
| 4 | Difícil | 3.0 m |
| 5 | Máximo | — |

## Licença

Este projeto está sob a licença MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.
