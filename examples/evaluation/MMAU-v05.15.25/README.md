## MMAU-v05.15.25 evaluation 

DeSTA2.5-Audio ❤️ 
Sound: 66.83333333333333
speech: 71.94194885970975
music: 57.099999999999994
Avg: 65.21250281088375


| Name                  | Size | Sound | Music | Speech | Avg   |
| --------------------- | ---- | ----- | ----- | ------ | ----- |
| Audio Flamingo 3    | 8.2B | 75.83 | 66.77 | 66.97  | 72.42 |
| Qwen2.5-Omni       | 8.2B | 76.77 | 65.90 | 68.90  | 71.00 |
| Gemini 2.5 Pro     | -    | 70.63 | 68.26 | 72.67  | 69.36 |
| Gemini 2.5 Flash      | -    | 69.50 | 65.57 | 68.27  | 67.39 |
| Gemini 2.0 Flash      | -    | 68.93 | 65.27 | 72.87  | 67.03 |
| **DeSTA2.5-Audio** ❤️     | 8 B | 66.83 | 57.09 |  71.94 | 65.21 |
| Kimi-Audio            | 8.2B | 70.70 | 66.77 | 56.57  | 64.40 |
| Audio Reasoner        | 8.2B | 67.27 | 69.16 | 62.53  | 63.78 |
| Phi-4-multimodal      | 5.5B | 62.67 | 64.37 | 63.80  | 62.81 |
| Gemini 2.5 Flash Lite | -    | 62.50 | 63.47 | 67.47  | 61.61 |
| Audio Flamingo 2      | 3B   | 68.13 | 70.96 | 44.87  | 61.06 |
| GPT-4o Audio          | -    | 63.20 | 56.29 | 69.33  | 60.82 |
| Qwen2-Audio-Instruct  | 7B   | 61.17 | 56.29 | 55.37  | 57.40 |
| Gemma 3n              | 4B   | 50.27 | 56.89 | 62.13  | 55.20 |
| Gemma 3n              | 2B   | 47.47 | 52.10 | 57.07  | 52.60 |
| GPT-4o mini Audio     | -    | 49.67 | 39.22 | 67.47  | 51.03 |



| Name                  | Size | Sound | Music | Speech | Avg   |
| --------------------- | ---- | ----- | ----- | ------ | ----- |
| Audio Flamingo 3    | 8.2B | 79.58 | 74.47 | 66.37  | 73.30 |
| Qwen2.5-Omni       | 8.2B | 78.10 | 67.33 | 70.60  | 71.50 |
| Gemini 2.5 Pro      | -    | 75.08 | 64.77 | 71.47  | 71.60 |
| Gemini 2.5 Flash      | -    | 73.27 | 69.40 | 76.58  | 71.80 |
| Gemini 2.0 Flash      | -    | 71.17 | 59.30 | 75.08  | 70.50 |
| Kimi-Audio            | 8.2B | 75.68 | 65.93 | 62.16  | 68.20 |
| Audio Reasoner        | 8.2B | 67.87 | 61.53 | 66.07  | 67.70 |
| DeSTA2.5-Audio ❤️     | 8 B | 71.17 | 70.57 | 56.29 | 66.00 |
| Phi-4-multimodal      | 5.5B | 65.47 | 61.97 | 67.27  | 65.70 |
| Gemini 2.5 Flash Lite | -    | 63.06 | 54.87 | 72.07  | 66.20 |
| Audio Flamingo 2      | 3B   | 71.47 | 70.20 | 44.74  | 62.40 |
| GPT-4o Audio          | -    | 64.56 | 49.93 | 66.67  | 62.50 |
| Qwen2-Audio-Instruct  | 7B   | 67.27 | 55.67 | 55.26  | 59.60 |
| Gemma 3n              | 4B   | 55.86 | 53.20 | 61.26  | 58.00 |
| Gemma 3n              | 2B   | 51.35 | 51.63 | 52.22  | 51.69 |
| GPT-4o mini Audio     | -    | 50.75 | 35.97 | 69.07  | 53.00 |




## Run

```shell
# MMAU test-mini
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/.cache python inference_desta25_audio.py --data_root /lab/DeSTA2.5-Audio/my_data -i ./MMAU-051525/data/mmau-test-mini.json --model_id desta25

# MMAU test
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/.cache python inference_desta25_audio.py --data_root /lab/DeSTA2.5-Audio/my_data -i ./MMAU-051525/data/mmau-test.json --model_id desta25
```

```shell
python mmau_evaluate.py --input /lab/DeSTA2.5-Audio/examples/evaluation/MMAU/results/mmau-test-mini_results.json
```


```
******************************
Task-wise Accuracy:
sound : 70.57% over 333 samples
music : 56.29% over 334 samples
speech : 71.17% over 333 samples
******************************
```
