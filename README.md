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

## Changelog

- 231108: Intent generation, conversation simulation 구현함. gpt3.5 를 활용하는데 퀄리티가 충분히 만족스럽진 않아서 gpt4 를 enable하면 좋겠음. 기본적인 플로우를 작성해서 vector embedding 기반 navigation과 비교할 수 있도록 구성했음. 자잘한 버그를 해결한 후에 수천개 짜리 test case는 이걸로 만들되 수백개 정도의 human-labeled test set도 있으면 좋을 것 같음.
- 231115: `gpt-4-turbo-1106` model 뚫음. 더 퀄리티 높은 intent 및 label generation. evaluate는 여전히 잘 안 돌아감. 아마 single q-a 정도만 해야될 것 같음. 텍스트에 emoji나 unicode가 있으면 문제가 많은 것 같아서 chatbot dataset을 한 번 cleasing 해야 할 듯. Vector embedding 고르고 similarity를 계산한 후, gpt model의 응답 similarity stats와 랜덤 stats를 비교하는 것으로 정답 label의 유효함을 보일 수 있음.
- 231116: chatbot cleasing 해둠. gpt-4-turbo는 FnCall에 답이 없고 gpt-4를 쓰기에는 좀 부담되어서 3.5-turbo로 하기로 함. jobs-homepage와 lead-homepage에 대해 q&a pair 데이터 뽑아둠.
