"""
MMAU Evaluation Inference Script for DeSTA2.5-Audio

This script runs inference on the MMAU benchmark using the DeSTA2.5-Audio model.
"""
import argparse
import json
import os
from tqdm import tqdm
from desta import DeSTA25AudioModel


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run MMAU evaluation with DeSTA2.5-Audio")
    parser.add_argument("-i", "--input_path", type=str, required=True,
                        help="Path to MMAU input JSON file")
    parser.add_argument("--model_id", type=str, default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
                        help="Model ID or path to load")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing audio files")
    parser.add_argument("-o", "--output_path", type=str, default="results",
                        help="Output filename (without extension)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run MMAU inference."""
    # Load model
    model = DeSTA25AudioModel.from_pretrained(args.model_id)
    model.to("cuda")
    model.eval()

    # Load MMAU data
    with open(args.input_path, "r") as f:
        data = json.load(f)

    # Run inference
    results = []
    system_prompt = 'Focus on the audio clips and instructions. Put your answer in the format "The correct answer is: "___" ".'

    for item in tqdm(data, desc="Processing"):
        audio_path = os.path.join(
            args.data_root,
            item["audio_id"].replace("./", "", 1)
        )

        # Build question with choices
        question = f"{item['question']} Choose from the following options: "
        choices = item["choices"]
        for i, option in enumerate(choices):
            question += f'"{option}"'
            if i == len(choices) - 2:
                question += " or "
            elif i < len(choices) - 1:
                question += ", "

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n\n{question}",
                "audios": [{"audio": audio_path}]
            }
        ]

        outputs = model.generate(messages=messages, max_new_tokens=512, do_sample=False)
        response = outputs.text[0]

        item["messages"] = messages
        item["model_output"] = response
        item["model_prediction"] = response.replace("The correct answer is: ", "").strip()
        results.append(item)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_file = f"results/{args.output_path}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)