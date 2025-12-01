![](https://github.com/user-attachments/assets/caf87d07-e8c3-48c9-814f-1d7dc83b4e50)

[📑 Paper](https://arxiv.org/abs/2507.02768) | [👩‍💻 Github](https://github.com/kehanlu/DeSTA2.5-Audio) | [🤗 Model](https://huggingface.co/collections/DeSTA-ntu/desta25-audio-686a6b9e71afd92e1dd87486) | [🤗 Dataset](https://huggingface.co/datasets/DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct)

**DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment**
> **Self-generated data is what you need for developing general-purpose LALMs!**

- 🧪 **A new training framework** ([read the paper](https://arxiv.org/abs/2507.02768))  
  - Highly scalable and efficient without task-specific instruction-tuning data  
  - Preserves language ability and avoids catastrophic forgetting  
  - Comprehensive studies on data quality in LALM development  
- 📦 **Open resources for the community**  
  - Model checkpoints and Training scripts
  - DeSTA-AQA5M dataset (5M audio-text pairs from 7,000 hours of audio)  


## ✨ News / Change logs
- 🔥 2025/12/01: **Training refactor**: Migrated to HuggingFace Transformers Trainer with Adafactor optimizer. Added comprehensive documentation and unit tests.
- 🚧 *Coming soon*: vLLM-based data construction script, detailed finetuning tutorials
- 2025/07/23: Released **training scripts**. Now you can train your own DeSTA-Audio. [📘 Training README](docs/train.md)
- 2025/07/21: Released **DeSTA-AQA-5M** dataset! [📘 Dataset README](docs/dataset.md) [🤗 DeSTA-AQA5M](https://huggingface.co/datasets/DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct)
- 2025/07/10: Inference code and model checkpoints are live! [🤗 DeSTA2.5-Audio](https://huggingface.co/collections/DeSTA-ntu/desta25-audio-686a6b9e71afd92e1dd87486)
- 2025/07/03: DeSTA2.5-Audio paper is on arXiv! [📄 arXiv:2507.02768](https://arxiv.org/abs/2507.02768)

## 📄 Documents

| Document       | Description                          |
|----------------|--------------------------------------|
| [Quickstart](#quickstart) | Quickly set up and run the DeSTA-Audio model. | 
| [docs/train.md](docs/train.md)    | Instructions and scripts for training the DeSTA-Audio model. |
| [docs/dataset.md](docs/dataset.md)  | Information about DeSTA-AQA5M           |
| [docs/evaluation_tips.md](docs/evaluation_tips.md) | Tips for evaluating DeSTA2.5-Audio |


## 🧐 Architecture

![](https://github.com/user-attachments/assets/f89dce86-2942-4644-aee5-a40ab4129328)


## 🚀Quickstart

### Installation
```shell
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
cd DeSTA2.5-Audio
pip install -e .
```

### Basic Usage
```python
from desta import DeSTA25AudioModel

# Load the model from Hugging Face
model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
model.to("cuda")

# Run inference with audio input
messages = [
    {
        "role": "system",
        "content": "Focus on the audio clips and instructions."
    },
    {
        "role": "user",
        "content": "<|AUDIO|>\nDescribe this audio.",
        "audios": [{
            "audio": "/path/to/audio.wav",  # Path to your audio file
            "text": None
        }]
    }
]

outputs = model.generate(
    messages=messages,
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    max_new_tokens=512
)

print(outputs.text)
```

## 📂 Dataset
See [docs/dataset.md](docs/dataset.md) for more details.
|         | Response Generated From | HuggingFace ID                                                                                                                     | Preview                                                                                      |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| DeSTA-AQA5M | Llama3.1-8B-Instruct    | [DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct](https://huggingface.co/datasets/DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct) | [🔍](https://huggingface.co/datasets/DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct/viewer) |

## 🚆 Training & Finetuning

**Latest Update**: The training pipeline has been refactored to use **HuggingFace Transformers Trainer** with **Adafactor optimizer**, replacing the previous PyTorch Lightning and Apex dependencies.

### Key Features
- ✅ **HuggingFace Trainer**: Industry-standard training framework
- ✅ **Adafactor Optimizer**: Memory-efficient adaptive learning rate optimizer
- ✅ **Whisper Large V3 Turbo**: Support for the latest Whisper model variant
- ✅ **Comprehensive Documentation**: Type hints and docstrings throughout the codebase
- ✅ **Unit Tests**: Verified components with pytest

### Quick Start

```bash
# Train with default configuration (debug mode)
cd examples/train
python train_desta.py --config-name desta25_debug +dataset=debug exp_dir=/tmp/desta_debug

# Train with full configuration
python train_desta.py --config-name desta25_llama31-8B_Qformer6L +dataset=DestaAQA-5M exp_dir=/path/to/output
```

### Configuration

Training configurations are in `examples/train/config/`:
- `desta25_debug.yaml`: For quick testing with small models
- `desta25_llama31-8B_Qformer6L.yaml`: Full training configuration

Key parameters:
- `model.encoder.model_id`: `openai/whisper-large-v3` or `openai/whisper-large-v3-turbo`
- `optim`: Adafactor optimizer (configured automatically via HF Trainer)
- `trainer`: Training settings (devices, epochs, precision, etc.)

See [docs/train.md](docs/train.md) for detailed training instructions.

### Example Training Script

```bash
bash examples/train/train_example.sh
```

## 📚 Citation
```bibtex
@article{lu2025desta25Audio,
  title={DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment},
  author={Lu, Ke-Han and Chen, Zhehuai and Fu, Szu-Wei and Yang, Chao-Han Huck and Huang, Sung-Feng and Yang, Chih-Kai and Yu, Chee-En and Chen, Chun-Wei and Chen, Wei-Chih and Huang, Chien-yu and others},
  journal={arXiv preprint arXiv:2507.02768},
  year={2025}
}

@inproceedings{lu2025developing,
  title={Developing instruction-following speech language model without speech instruction-tuning data},
  author={Lu, Ke-Han and Chen, Zhehuai and Fu, Szu-Wei and Yang, Chao-Han Huck and Balam, Jagadeesh and Ginsburg, Boris and Wang, Yu-Chiang Frank and Lee, Hung-yi},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

@inproceedings{lu24c_interspeech,
  title     = {DeSTA: Enhancing Speech Language Models through Descriptive Speech-Text Alignment},
  author    = {Ke-Han Lu and Zhehuai Chen and Szu-Wei Fu and He Huang and Boris Ginsburg and Yu-Chiang Frank Wang and Hung-yi Lee},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4159--4163},
  doi       = {10.21437/Interspeech.2024-457},
  issn      = {2958-1796},
}
```



## 👥 Contributors
Ke-Han Lu, Zhehuai Chen, Szu-Wei Fu, Chao-Han Huck Yang, Sung-Feng Huang, Chih-Kai Yang, Chee-En Yu, Chun-Wei Chen, Wei-Chih Chen, Chien-yu Huang, Yi-Cheng Lin, Yu-Xiang Lin, Chi-An Fu, Chun-Yi Kuan, Wenze Ren, Xuanjun Chen, Wei-Ping Huang, En-Pei Hu, Tzu-Quan Lin, Yuan-Kuei Wu, Kuan-Po Huang, Hsiao-Ying Huang, Huang-Cheng Chou, Kai-Wei Chang, Cheng-Han Chiang, Boris Ginsburg, Yu-Chiang Frank Wang, Hung-yi Lee
