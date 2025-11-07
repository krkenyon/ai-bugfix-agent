import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class BugFixModel:
    """
    Thin wrapper around a causal language model for code fixing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        max_input_tokens: int = 512,
    ):
        if device is None:
            # use GPU if available - much faster for inference
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Format as chat for Qwen3
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 1) Keep only newly generated tokens (strip the prompt)
        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 2) Drop any <think>...</think> sections
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 3) If there's a fenced code block, keep only its content
        fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()

        # 4) From whatever remains, extract the first contiguous block of code
        #    starting at a line that looks like Python, and stop when it turns into prose.
        lines = text.splitlines()
        code_lines = []
        started = False

        for line in lines:
            stripped = line.strip()

            if not started:
                # Look for the first line that looks like real code
                if stripped.startswith(("def ", "class ", "import ", "from ")):
                    started = True
                    code_lines.append(line)
                # ignore everything until then
            else:
                # Once in code mode:
                if stripped == "":
                    # blank lines inside code are fine
                    code_lines.append(line)
                    continue

                # Lines that still look like Python code are allowed
                if (
                    stripped.startswith(("def ", "class ", "import ", "from ", "@", "#"))
                    or line.startswith((" ", "\t"))  # indented code
                ):
                    code_lines.append(line)
                else:
                    # As soon as it looks like natural language, stop.
                    break

        if code_lines:
            text = "\n".join(code_lines).strip()

        return text
