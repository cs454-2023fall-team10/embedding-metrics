# embedding-metrics

Attempts to evaluate quality of vector embeddings

## Generate intents powered by OpenAI

```bash
python3 intent.py <chatbot-filename> <output-filename> <num-intents>
```

## Simulate conversations powered by OpenAI

```bash
python3 label.py <chatbot-filename> <intent-filename> <num-conversations> <output-filename>
```

## Evaluate vector embeddings

1. Activate environment for the embedding models to evaluate.
2. Uncomment the lines L146-148 for embedding models to evaluate.
3.

```bash
python evaluate.py <sample-prompt-filename>
```

## Evaluate result (Prediction Loss)

| Embedding     | Loss (`jobs-homepage`) | Accuracy (`jobs-homepage`) | Loss (`lead-homepage`) | Accuracy (`lead-homepage`) |
| ------------- | ---------------------- | -------------------------- | ---------------------- | -------------------------- |
| Random        |               644.532  |                      0.242 |                552.667 |                      0.344 |
| fasttext      |               649.972  |                      0.278 |                514.598 |                      0.616 | 
| openai        |               620.134  |                      0.348 |                520.351 |                      0.848 | 
| pororo        |               614.527  |                      0.380 |                517.770 |                      0.524 | 
| sentence_bert |               586.885  |                      0.472 |                489.947 |                      0.730 | 

## Changelog

- 231108: Intent generation, conversation simulation 구현함. gpt3.5 를 활용하는데 퀄리티가 충분히 만족스럽진 않아서 gpt4 를 enable하면 좋겠음. 기본적인 플로우를 작성해서 vector embedding 기반 navigation과 비교할 수 있도록 구성했음. 자잘한 버그를 해결한 후에 수천개 짜리 test case는 이걸로 만들되 수백개 정도의 human-labeled test set도 있으면 좋을 것 같음.
- 231115: `gpt-4-turbo-1106` model 뚫음. 더 퀄리티 높은 intent 및 label generation. evaluate는 여전히 잘 안 돌아감. 아마 single q-a 정도만 해야될 것 같음. 텍스트에 emoji나 unicode가 있으면 문제가 많은 것 같아서 chatbot dataset을 한 번 cleasing 해야 할 듯. Vector embedding 고르고 similarity를 계산한 후, gpt model의 응답 similarity stats와 랜덤 stats를 비교하는 것으로 정답 label의 유효함을 보일 수 있음.
- 231116: chatbot cleasing 해둠. gpt-4-turbo는 FnCall에 답이 없고 gpt-4를 쓰기에는 좀 부담되어서 3.5-turbo로 하기로 함. jobs-homepage와 lead-homepage에 대해 q&a pair 데이터 뽑아둠.
- 231118: fasttext, openai, pororo, sentence_bert 4개 model에 대해 정확도와 cross entropy loss 측정했음. openai ada model과 bert를 사용하는 것이 좋아보임. 일단 두 가지 모델을 사용해서 sentence similarity를 측정하는 코드를 작성해둠.