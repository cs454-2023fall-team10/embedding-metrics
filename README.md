# embedding-metrics

Attempts to evaluate quality of vector embeddings

## Generate intents powered by OpenAI

```bash
python3 intent.py <chatbot-filename> <output-filename> <num-intents>
```

## Simulate conversations powered by OpenAI

```bash
python3 evaluate.py <chatbot-filename> <intent-filename> <num-conversations>
```

## Changelog

- 231108: Intent generation, conversation simulation 구현함. gpt3.5 를 활용하는데 퀄리티가 충분히 만족스럽진 않아서 gpt4 를 enable하면 좋겠음. 기본적인 플로우를 작성해서 vector embedding 기반 navigation과 비교할 수 있도록 구성했음. 자잘한 버그를 해결한 후에 수천개 짜리 test case는 이걸로 만들되 수백개 정도의 human-labeled test set도 있으면 좋을 것 같음.
